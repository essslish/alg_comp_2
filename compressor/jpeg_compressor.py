import struct
from typing import Tuple, List

import numpy as np
from PIL import Image

from compressor.base import Compressor
from compressor.ha import HACompressor
from compressor.rle import RLECompressor
from functions.DCDifferentialCodec import DCDifferentialCodec
from functions.DCT2D import DCT2D
from functions.RGBToYCbCr import RGBToYCbCr
from functions.blocks import BlockSplitter
from functions.downsampler import Downsampler
from functions.qantizer import Quantizer
from functions.zigzag import ZigZag
from utils.logger import logger


class JPEGCompressor(Compressor):
    HEADER_FMT = ">HHBB"  # width, height (uint16), block_size(uint8), quality(uint8)

    def __init__(
            self,
            quality: int = 75,
            block_size: int = 8,
            subsample_xy: Tuple[int, int] = (2, 2),
            fill_value: int = 0
    ):
        if not (0 <= quality <= 100):
            raise ValueError("quality должно быть в диапазоне [0…100]")
        self.quality = quality
        self.block_size = block_size
        self.sub_y, self.sub_x = subsample_xy
        self.fill_value = fill_value

        # Инициализируем вспомогательные модули
        self.dct = DCT2D(block_size)
        self.quant_tables = Quantizer.get_quant_tables(quality)
        self.splitter = BlockSplitter(block_size, fill_value)
        self.downsampler = Downsampler(self.sub_y, self.sub_x, fill_value)
        self.zigzag = ZigZag(block_size)
        self.dc_codec = DCDifferentialCodec()
        self.rle = RLECompressor()
        self.huffman = HACompressor()

        logger.debug(f"Initialized JPEGCompressor: quality={quality}, block_size={block_size}, "
                     f"subsample={subsample_xy}, fill_value={fill_value}")

    def compress(self, img: Image.Image) -> bytes:
        # 1. RGB → bytes
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        rgb_bytes = img_rgb.tobytes()
        logger.debug(f"[1] RGB bytes: length={len(rgb_bytes)}, image_size={width}x{height}")

        # 2. RGB → YCbCr
        ycbcr_bytes = RGBToYCbCr.convert(rgb_bytes)
        arr = np.frombuffer(ycbcr_bytes, dtype=np.uint8).reshape((height, width, 3))
        Y, Cb, Cr = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        logger.debug(f"[2] YCbCr shapes — Y:{Y.shape}, Cb:{Cb.shape}, Cr:{Cr.shape}")

        # 3. Downsampling
        Cb_ds, _ = self.downsampler.downsample(Cb)
        Cr_ds, _ = self.downsampler.downsample(Cr)
        logger.debug(f"[3] Downsampled Cb:{Cb_ds.shape}, Cr:{Cr_ds.shape}")

        # 4. Split into blocks
        blocks_Y = self.splitter.convert(Y.tobytes(), width, height)
        h_cb, w_cb = Cb_ds.shape
        Cb_u8 = np.round(Cb_ds).astype(np.uint8)
        blocks_Cb = self.splitter.convert(Cb_u8.tobytes(), w_cb, h_cb)
        h_cr, w_cr = Cr_ds.shape
        Cr_u8 = np.round(Cr_ds).astype(np.uint8)
        blocks_Cr = self.splitter.convert(Cr_u8.tobytes(), w_cr, h_cr)
        logger.debug(f"[4] Blocks count — Y:{len(blocks_Y)}, Cb:{len(blocks_Cb)}, Cr:{len(blocks_Cr)}")

        # 5. DCT → Quant → ZigZag
        qt_Y, qt_C = self.quant_tables
        logger.debug(f"[5] Quant tables shapes — Y:{qt_Y.shape}, Cb/Cr:{qt_C.shape}")

        def _process(blocks: List[bytes], table: np.ndarray, name: str) -> List[np.ndarray]:
            out = []
            for i, buf in enumerate(blocks):
                mat = np.frombuffer(buf, dtype=np.uint8).astype(np.float64).reshape(
                    (self.block_size, self.block_size)) - 128
                C = self.dct.forward(mat)
                Q = Quantizer.quantize(C, table)
                zz = self.zigzag.encode(Q)
                out.append(zz)
                if i == 0:
                    logger.debug(f"    [{name}][0] DCT C[0,0]={C[0, 0]:.2f}, Q[0,0]={Q[0, 0]}, ZigZag[0]={zz[0]}")
            return out

        zz_Y = _process(blocks_Y, qt_Y, "Y")
        zz_Cb = _process(blocks_Cb, qt_C, "Cb")
        zz_Cr = _process(blocks_Cr, qt_C, "Cr")

        # 6. Gather all
        all_blocks = zz_Y + zz_Cb + zz_Cr
        logger.debug(f"[6] Total zigzag blocks: {len(all_blocks)}")

        # 7. DC & AC split
        dc_vals = [int(blk[0]) for blk in all_blocks]
        # AC: blk[1:] — np.ndarray длины bs*bs-1, dtype=int16
        ac_vals = b"".join(blk[1:].astype(np.int16).tobytes() for blk in all_blocks)
        logger.debug(f"[7] DC count={len(dc_vals)}, AC total_bytes={len(ac_vals)}")

        # 8. DC diff + RLE AC
        dc_diff = self.dc_codec.encode(dc_vals)
        dc_bytes = np.array(dc_diff, dtype=np.int16).tobytes()
        ac_rle = self.rle.compress(ac_vals)
        logger.debug(f"[8] DC diff_len={len(dc_diff)}, DC_bytes={len(dc_bytes)}, AC_RLE={len(ac_rle)}")

        # 9. Huffman
        payload = self.huffman.compress(dc_bytes + ac_rle)
        logger.debug(f"[9] Huffman payload length={len(payload)}")

        # 10. Header
        header = struct.pack(self.HEADER_FMT, width, height, self.block_size, self.quality)
        header += struct.pack(">HH", w_cb, h_cb)
        logger.debug(f"[10] Header length={len(header)} — total output={len(header) + len(payload)}")

        return header + payload

    def decompress(self, data: bytes) -> Image.Image:
        """
        Декомпрессия собственного формата в PIL.Image.
        """
        # 1. Разбор заголовка
        hdr_sz = struct.calcsize(self.HEADER_FMT) + struct.calcsize(">HH")
        header = data[:hdr_sz]
        payload = data[hdr_sz:]
        w, h, bs, q = struct.unpack(self.HEADER_FMT,
                                    header[:struct.calcsize(self.HEADER_FMT)])
        ds_w, ds_h = struct.unpack(">HH", header[-4:])
        logger.debug(f"[D1] Header — w={w}, h={h}, bs={bs}, q={q}, ds={ds_w}×{ds_h}")

        # 2. Реинициализируем модули
        qt_Y, qt_C = Quantizer.get_quant_tables(q)
        dct = DCT2D(bs)
        splitter = BlockSplitter(bs, self.fill_value)
        downsampler = Downsampler(self.sub_y, self.sub_x, self.fill_value)
        zigzag = ZigZag(bs)
        dc_codec = DCDifferentialCodec()
        rle = RLECompressor()
        huffman = HACompressor()

        # 3. Huffman-декодирование
        combined = huffman.decompress(payload)
        byY, bxY = (w + bs - 1) // bs, (h + bs - 1) // bs
        nY = byY * bxY
        byC, bxC = (ds_w + bs - 1) // bs, (ds_h + bs - 1) // bs
        nC = byC * bxC
        total = nY + 2 * nC

        dc_bytes_len = total * 2
        dc_bytes = combined[:dc_bytes_len]
        ac_rle = combined[dc_bytes_len:]
        logger.debug(f"[D3] Blocks Y={nY}, Cb/Cr={nC}, total={total}; "
                     f"DC_bytes={len(dc_bytes)}, AC_RLE={len(ac_rle)}")

        # 4. DC diff + RLE AC
        dc_diff = np.frombuffer(dc_bytes, dtype=np.int16).tolist()
        dc_vals = dc_codec.decode(dc_diff)
        ac_vals = rle.decompress(ac_rle)
        logger.debug(f"[D4] DC count={len(dc_vals)}, AC bytes={len(ac_vals)}")

        # 5. Reconstruct zigzag-блоки
        all_blocks = []
        ac_block_bytes = (bs * bs - 1) * 2
        off = 0

        for i in range(total):
            # каждый блок: первые 2 байта — DC (мы их подставим), затем AC
            ac_chunk = ac_vals[off: off + ac_block_bytes]
            # читаем AC-коэффициенты как int16
            zz_ac = np.frombuffer(ac_chunk, dtype=np.int16)

            # собираем полный zigzag-массив длины bs*bs
            blk = np.empty(bs * bs, dtype=np.int16)
            blk[0] = dc_vals[i]
            blk[1:] = zz_ac

            all_blocks.append(blk)
            off += ac_block_bytes
        logger.debug(f"[D5] Reconstructed zigzag blocks: {len(all_blocks)}")

        # 6. Inverse DCT+dequant
        def _restore(zblks, table, name):
            mats = []
            for idx, zz in enumerate(zblks):
                Q = zigzag.decode(zz)
                C = Quantizer.dequantize(Q, table)
                mat = dct.inverse(C) + 128
                mats.append(np.clip(mat, 0, 255).astype(np.uint8))
                if idx == 0:
                    logger.debug(f"    [{name}][0] Q00={Q[0, 0]}, pix00={mats[-1][0, 0]}")
            return mats

        out_Y = _restore(all_blocks[:nY], qt_Y, "Y")
        out_Cb = _restore(all_blocks[nY:nY + nC], qt_C, "Cb")
        out_Cr = _restore(all_blocks[nY + nC:nY + 2 * nC], qt_C, "Cr")
        logger.debug(f"[D6] Restored blocks — Y:{len(out_Y)}, Cb:{len(out_Cb)}, Cr:{len(out_Cr)}")

        # 7. Merge & upsample
        def merge(blocks: List[np.ndarray], ow: int, oh: int) -> np.ndarray:
            raw = splitter.inverse([b.tobytes() for b in blocks], ow, oh)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((oh, ow))
            return arr

        Yc = merge(out_Y, w, h)
        Cbc = merge(out_Cb, ds_w, ds_h)
        Crc = merge(out_Cr, ds_w, ds_h)
        Cb_up = downsampler.upsample(Cbc, (h, w))
        Cr_up = downsampler.upsample(Crc, (h, w))
        logger.debug(f"[D7] Merged & upsampled — Y:{Yc.shape}, Cb:{Cb_up.shape}, Cr:{Cr_up.shape}")

        # 8. Final RGB
        merged = np.stack([Yc, Cb_up, Cr_up], axis=2).tobytes()
        rgb = RGBToYCbCr.inverse(merged)
        img = Image.frombytes("RGB", (w, h), rgb)
        logger.debug("[D8] Decompression complete")
        return img
