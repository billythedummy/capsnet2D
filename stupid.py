import subprocess

if __name__ == "__main__":
    epochs = 600
    i = 0
    lr_decay = 0.95
    lr = 0.001 #half of original
    while i < epochs:
        subprocess.call("python __main__.py --batch_size=4 --from_saved=1 --epochs=1 --lr="+str(lr), shell=True)
        i += 1
        lr *= lr_decay
