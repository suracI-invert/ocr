from src.models.lit_module import OCRLitModule
from src.models.model import Net
from src.models.tokenizer import Tokenizer
from src.data.datamodule import OCRDataModule
from src.data.components.collator import Collator
from src.utils.transforms import Resize, ToTensor, SwinAugmenter, AlbumentationsWithPadding

from torch import set_float32_matmul_precision
from torchvision.transforms import Compose
from lightning import Trainer

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    set_float32_matmul_precision('medium')

    CKPT_PATH = './weights/vietocr_resnet_epoch=06_val_cer=0.13.ckpt'

    # cnn_args = {
    #     'cnn_arch': 'vgg',
    #     'weights': 'IMAGENET1K_V1',
    #     'ss': [
    #         [2, 2],
    #         [2, 2],
    #         [2, 1],
    #         [2, 1],
    #         [1, 1]
    #     ],
    #     'ks': [
    #         [2, 2],
    #         [2, 2],
    #         [2, 1],
    #         [2, 1],
    #         [1, 1]
    #     ],
    #     'hidden': 256
    # }

    cnn_args = {
        'cnn_arch': 'resnet50',
        'ss': [
            [2, 2],
            [2, 1],
            [2, 1],
            [2, 1],
            [1, 1]
        ],
        'hidden': 256
    }

    
    swin_args = {
        'hidden': 256,
        'dropout': 0.2,
        'pretrained': 'microsoft/swin-tiny-patch4-window7-224'
    }

    trans_args = {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "max_seq_length": 512,
        "pos_dropout": 0.2,
        "trans_dropout": 0.1
    }

    tokenizer = Tokenizer(trans_args['max_seq_length'])

    collator = Collator()

    dataModule = OCRDataModule(
        data_dir= './data/', map_file= 'train_annotation.txt',
        test_dir= './data/new_public_test',
        tokenizer= tokenizer,
        train_val_split= [100_000, 3_000],
        batch_size= 64,
        num_workers= 6,
        pin_memory= True,
        transforms= AlbumentationsWithPadding((224, 112)),
        collate_fn= collator,
        sampler= None
    )

    dataModule.setup()

    test_loader = dataModule.test_dataloader()

    net = Net(len(tokenizer.chars), 'cnn', cnn_args, trans_args)

    model = OCRLitModule.load_from_checkpoint(net= net, tokenizer= tokenizer,
        checkpoint_path= CKPT_PATH,
    )

    trainer = Trainer(
        accelerator= 'gpu',
        precision= '16-mixed',
        logger= False
    )

    res = trainer.predict(model, test_loader)


    filenames = []
    predictions = []
    probabilities = []
    prediction_path = './data/prediction.txt'
    with open(prediction_path, 'wt', encoding= 'utf8') as f:
        count = 0
        for pb in tqdm(res, desc= 'Accessing batch', position= 0):
            for p in tqdm(pb, desc= 'Writing to: ' + prediction_path, position= 1, leave= False):
                line = p['filename'] + '\t' + p['prediction'] + '\n'
                f.write(line)
                filenames.append(p['filename'])
                predictions.append(p['prediction'])
                probabilities.append(p['probability'].item())
                count += 1

    print(f'Predicted total of {count} images')

    csv_path = './data/predictions_prob.csv'
    print(f'Exporting csv file: {csv_path}')
    pd.DataFrame(
        {
            'filename': filenames,
            'prediction': predictions,
            'probability': probabilities
        }
    ).to_csv(csv_path, index= False)
    print('DONE!')