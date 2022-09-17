from transformers import AutoTokenizer
from Model.utils import dfs_set_checkpoint


from configs.CoordinateConfig import CoordinateGTPContrastiveConfig as Config


splits = [slice(0, 391), slice(391, 782), slice(782, 1173), slice(1173, 1564), slice(1564, 1955)]


if __name__ == "__main__":
    # for n_samples in [16, 32, 48, 64, 128]:
    # for alpha in [0.01, 0.1, 1, 10, 100, 250]:
    # for alpha in [500, 750, 1000, 2000, 10000, 100000]:
    # for n_samples in [2, 4, 8, 16, 32, 48, 64, 128]:
    # for n_samples in [128, 256, 512, 1024, 1536, 2048]:
    for i in range(5):
        # args = Config(model_name=f"{Config.model_name}_{alpha}_{i}", device="cuda:0",
        #               cross_weight=alpha,
        # args = Config(model_name=f"{Config.model_name}_k{n_samples}_{i}", device="cuda:2",
        #               n_samples=n_samples, temperature=n_samples,
        args = Config(model_name=f"{Config.model_name}_{i}", device="cuda:3",
                      train_val_split=(splits[:i] + splits[i+1:], splits[i:i+1]))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir="cache")
        model = args.Model(args).cuda()
        optimizer = args.optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainer = args.Trainer(args, model, tokenizer, optimizer)
        model.load_pretrained(args) if args.load_pretrained else None
        dfs_set_checkpoint(model) if args.gradient_checkpoint else None
        trainer.train()
