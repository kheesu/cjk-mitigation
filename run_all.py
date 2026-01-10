from src.finetune import Experiments 

def main():
    e = Experiments('config.yml')
    e.finetune_all()

if __name__ == "__main__":
    main()
