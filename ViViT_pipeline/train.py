import torch

from utils.helper import *

from utils.configuration import CONFIG

CONFIG = CONFIG()


def main():
    print(f'#'*50)
    print('Training Started')
    print(f'#'*50)
    # f1scores = []
    # accuracy= []
    
    data_df = load_data()  

    image_processor, model = load_model(model_name=CONFIG.model_name ,cfg=CONFIG)
    
    train_loader, valid_loader = get_data_loader(data_df=data_df, image_processor=image_processor)
    
    optimizer = get_optimizer(parameters=model.parameters(), cfg=CONFIG)
    scheduler= get_scheduler(optimizer=optimizer)
    
    model, history, best_score, accuracy= training_loop(model, 
                                                        train_loader, 
                                                        valid_loader,
                                                        optimizer, 
                                                        scheduler,
                                                        num_epochs=CONFIG.num_epochs, 
                                                        patience=CONFIG.patience,
                                                        cfg= CONFIG
                                                        )
    
    return model, history


if __name__ == "__main__":
    main()