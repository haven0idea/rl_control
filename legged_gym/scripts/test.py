import wandb
wandb.login()
wandb.init(project="proxy-test")
wandb.log({"proxy_test": 1})
wandb.finish()
