import torch
from tqdm import tqdm


def get_representation(model, data):
    with torch.no_grad():
        value = model.encode_image(data)
    return value


def estimate_content(model, train_loader, lamda=1000): 
    original_representation = []
    intervene_representation = []
    for (x_orig, x_intervene, path), _ in tqdm(train_loader, desc="Loading training data"):
        z = get_representation(model, x_orig.cuda())
        z_intervene = get_representation(model, x_intervene.cuda())
        original_representation.append(z)
        intervene_representation.append(z_intervene)
    original_representation = torch.cat(original_representation)
    intervene_representation = torch.cat(intervene_representation)
    delta = original_representation - intervene_representation
    
    n_or_re = original_representation.shape[0]
    n_delta = delta.shape[0]
    M = original_representation.T @ original_representation / n_or_re - lamda * delta.T @ delta / n_delta
    eig_value, eig_vector = torch.linalg.eigh(M)
    return eig_vector 

def get_A_inv(configs, model, train_loader):
    # hold_out_dim is the number of dimensions to keep for content representation
    A_inv = estimate_content(model, train_loader, lamda=configs.lamda)[:, :configs.hold_out_dim]
    return A_inv  