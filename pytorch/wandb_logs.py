import wandb

n_layer = None

def wandb_init(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ml710-project-pipeline-parallelism",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "learning_rate_cosine_scheduler": args.eta_min,
            "architecture": "transformer-xl",
            "dataset": args.dataset,
            "epochs": 1
        },
        tags=["gpus - 4"]
    )

cols = [f'layer{i}' for i in range(12)]
cols.append('pipeline_time')

gpu_table = wandb.Table(columns=cols)
# table = wandb.Table(columns=["text", "true_text", "predicted_text"])


def log_gpu_table(layers, pipeline_time):
    # layers = [layer.to('cpu') for layer in layers]
    gpu_table.add_data(*layers, pipeline_time)


def log_table(text, true_text, predicted_text):
    text, true_text, predicted_text = text.to("cpu"), predicted_text.to("cpu"), true_text.to("cpu")
    table.add_data(text, true_text, predicted_text)