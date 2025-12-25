from .decoder_utils import *
from .encoder_utils import *


class MS2Edge(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, args=None):
        super().__init__()
        self.T = args.T
        self.D = 4
        k = 1
        self.down_channels = [32 * k, 64 * k, 64 * k, 128 * k, 256 * k, 512 * k]
        step_mode = args.step_mode
        backend = args.backend
        self.args = args

        self.plain_conv = SpikingPlainBlock(in_channel, 32 * k, 7, 3, step_mode, backend, args)
        self.encoder = get_model(args)

        self.skip1 = SkipBlock(64 * k, 64 * k, 3, 1, 1, step_mode, backend, args)
        self.skip2 = SkipBlock(128 * k, 128 * k, 1, 1, 0, step_mode, backend, args)
        self.skip3 = SkipBlock(256 * k, 256 * k, 1, 1, 0, step_mode, backend, args)
        self.skip4 = SkipBlock(512 * k, 512 * k, 1, 1, 0, step_mode, backend, args)

        self.bottleneck = nn.Sequential(
            MDSBlock(512 * k, 512 * k, step_mode=step_mode, backend=backend),
            MSBlock(512 * k, 512 * k, step_mode=step_mode, backend=backend),
        )

        self.decoder_layer5 = SpikingUpBlock(512 * k, 512 * k, 3, 1, 2, step_mode, backend, args)
        self.decoder_layer4 = SpikingUpBlock(512 * k, 256 * k, 3, 1, 2, step_mode, backend, args)
        self.decoder_layer3 = SpikingUpBlock(256 * k, 128 * k, 3, 1, 2, step_mode, backend, args)
        self.decoder_layer2 = SpikingUpBlock(128 * k, 64 * k, 3, 1, 2, step_mode, backend, args)
        self.decoder_layer1 = SpikingUpBlock(64 * k, 32 * k, 3, 1, 1, step_mode, backend, args)

        self.predict_depth4 = SpikingPredUpBlock(512 * k, out_channel, 7, 3, 8, step_mode=step_mode, backend=backend,
                                                 args=args)
        self.predict_depth3 = SpikingPredUpBlock(256 * k, out_channel, 7, 3, 4, step_mode=step_mode, backend=backend,
                                                 args=args)

        self.predict_depth2 = SpikingPredUpBlock(128 * k, out_channel, 7, 3, 2, step_mode=step_mode, backend=backend,
                                                 args=args)

        self.predict_depth1 = SpikingPredUpBlock(64 * k, out_channel, 7, 3, 1, step_mode=step_mode, backend=backend,
                                                 args=args)

        self.predict_depth0 = SpikingPredUpBlock(32 * k, out_channel, 7, 3, 1, step_mode=step_mode, backend=backend,
                                                 args=args)

    def forward(self, x):
        # 时序维度扩展
        x = x.repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]

        # 编码
        encode_down, encode_x = self.plain_conv(x)
        encode_down1, encode_down2, encode_down3, encode_down4, encode_down5 = self.encoder(encode_down)

        encode_x1 = self.skip1(encode_down1)
        encode_x2 = self.skip2(encode_down2)
        encode_x3 = self.skip3(encode_down3)
        encode_x4 = self.skip4(encode_down4)

        bottleneck_x = self.bottleneck(encode_down5)

        # 解码 + 融合
        decode_up4 = self.decoder_layer5(bottleneck_x)

        fusion4 = encode_x4 + decode_up4
        decode_up3 = self.decoder_layer4(fusion4)

        fusion3 = encode_x3 + decode_up3
        decode_up2 = self.decoder_layer3(fusion3)

        fusion2 = encode_x2 + decode_up2
        decode_up1 = self.decoder_layer2(fusion2)

        fusion1 = encode_x1 + decode_up1
        decode_up = self.decoder_layer1(fusion1)

        fusion = encode_x + decode_up

        # 输出预测
        pred_edge5 = self.predict_depth4(fusion4)

        pred_edge4 = self.predict_depth3(fusion3)

        pred_edge3 = self.predict_depth2(fusion2)

        pred_edge2 = self.predict_depth1(fusion1)

        pred_edge1 = self.predict_depth0(fusion)

        return pred_edge1, pred_edge2, pred_edge3, pred_edge4, pred_edge5
