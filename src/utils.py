import os
import config
from sklearn import metrics

def evaluate_auc(y_trues, y_preds):
    """ Evaluate predictions using AUC """
            
    # AUC metric
    try:
        auc = metrics.roc_auc_score(y_trues, y_preds)
    except:
        auc = 0.5

    return auc


def get_model(task, plane, cut, path):
    """ Get best model from directory for a particular task, plane & cut """

    assert task in config.TASKS
    assert plane in config.PLANES
    assert cut in config.SLICING

    # Get best model
    models = os.listdir(path)
    model_names = list(filter(lambda name: task in name and 
        plane in name and 
        'cut_' + cut in name, models))

    if len(model_names) == 0:
        return None
        # raise Exception('Model not found for task {}, plane {} & cut {}'.format(
        #     task, plane, cut))
    
    # Grab the model with the highest Validation AUC
    models_val_auc = { model_name: float(model_name.split('val_auc_')[1].split('_')[0]) 
        for model_name in model_names }
    # Max Validation AUC
    max_value = max(models_val_auc.values())
    # First model with that max value
    result = list(filter(lambda x: x[1] == max_value, models_val_auc.items()))[0]
    model_name = result[0]
    model_architecture = model_name.split('arch_')[1].split('_')[0]
    model_augment = model_name.split('augment_')[1].split('_')[0]
    augment_probability = float(model_name.split('augment-probability_')[1].split('.pth')[0])
    
    model_path = '{}/{}'.format(path, model_name)
    print('[UTILS] Model name "{}" for "{}", Val AUC "{}", Architecture: "{}", Augment: "{}" (prob {})'.format(
        model_name, plane, max_value, model_architecture, model_augment, augment_probability))

    return { 
        'model_path': model_path, 
        'model_name': model_name,
        'model_val_auc': max_value,
        'model_architecture': model_architecture,
        'model_augment': model_augment, 
        'augment_probability': augment_probability,
    }


def get_best_predictions(task, plane, cut, path):
    """ Get best predictions for a particular task, plane & cut """

    # Get best model
    preds = os.listdir(path)
    pred_names = list(filter(lambda name: task in name and 
        plane in name and 
        'cut_' + cut in name, preds))

    if len(pred_names) == 0:
        raise Exception('Model not found for task {}, plane {} & cut {}'.format(
            task, plane, cut))
    
    # Grab the model with the highest Validation AUC
    preds_val_auc = { pred_name: float(pred_name.split('val_auc_')[1].split('_')[0]) 
        for pred_name in pred_names }
    # Max Validation AUC
    max_value = max(preds_val_auc.values())
    # First model with that max value
    result = list(filter(lambda x: x[1] == max_value, preds_val_auc.items()))[0]
    pred_name = result[0]
    pred_architecture = pred_name.split('arch_')[1].split('_')[0]
    pred_augment = pred_name.split('augment_')[1].split('_')[0]
    augment_probability = float(pred_name.split('augment-probability_')[1].split('.pth')[0])
    
    pred_path = '{}/{}'.format(path, pred_name)
    print('[UTILS] Model name "{}" for "{}", Val AUC "{}", Architecture: "{}", Augment: "{}" (prob {})'.format(
        pred_name, plane, max_value, pred_architecture, pred_augment, augment_probability))

    return { 
        'pred_path': pred_path, 
        'pred_architecture': pred_architecture,
        'pred_augment': pred_augment, 
        'augment_probability': augment_probability,
    }


def load_bundle_model(task, path):
    """ Load Logistic Regression model """

    assert task in config.TASKS

    # Get model
    models = os.listdir(path)
    model_names = list(filter(lambda name: '.joblib' in name, models))
    
    models_val_auc = { model_name: float(model_name.split('val_auc_')[1].split('.')[0]) 
        for model_name in model_names }
    # Max Validation AUC
    max_value = max(models_val_auc.values())
    # First model with that max value
    result = list(filter(lambda x: x[1] == max_value, models_val_auc.items()))[0]
    model_name = result[0]

    model_path = '{}/{}'.format(path, model_name)

    print('[UTILS] Model name "{}" for "{}"'.format(model_name, task))

    return { 
        'model_path': model_path, 
    }
