from torch import bool, int64, is_tensor, tensor


class Tokenizer:
    def __init__(self):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = (
            'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'
            r"()*+,-./:;<=>?@[\]^_`{|}~ "
        )
        self.c2i = {c: i + 4 for i, c in enumerate(self.chars)}
        self.i2c = {i + 4: c for i, c in enumerate(self.chars)}

        self.i2c[0] = "<pad>"
        self.i2c[1] = "<s>"
        self.i2c[2] = "</s>"
        self.i2c[3] = "*"

    def encode(
        self,
        chars,
        padding: bool = False,
        max_length: int = 64,
        return_attn_mask=True,
        return_tensor: bool = False,
    ):
        ids = [self.go] + [self.c2i[c] for c in chars] + [self.eos]

        if padding:
            ids += [self.pad for _ in range(max_length - len(ids))]

        attn_mask = [False if i != 0 else True for i in ids]

        if return_tensor:
            ids = tensor(ids, dtype=int64)
            attn_mask = tensor(attn_mask, dtype=bool)

        if return_attn_mask:
            return {"input_ids": ids, "attention_masks": attn_mask}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_token=True):
        ids = ids.cpu().tolist() if is_tensor(ids) else ids

        if skip_special_token:
            return "".join([self.i2c[i] for i in ids if i not in [0, 1, 2, 3]])
        return "".join([self.i2c[i] for i in ids])

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, batch_ids, skip_special_token=True):
        return [self.decode(ids, skip_special_token) for ids in batch_ids]

    def __str__(self):
        return self.chars
