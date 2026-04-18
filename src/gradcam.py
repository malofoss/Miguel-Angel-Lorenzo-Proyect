import torch
import torch.nn.functional as F
import cv2
import numpy as np

class AgeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Conectar hooks para capturar datos internos
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, original_image_np):
        """
        Calcula el mapa de calor dado un tensor de entrada (ya normalizado) 
        y la imagen original de numpy para superponerlo.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Necesitamos que todo el gráfico guarde los gradientes
        input_tensor.requires_grad = True
        
        # Forward pass (predicción)
        class_prob, age_pred = self.model(input_tensor)
        
        # Backward pass usando la predicción de edad como el objetivo a explicar.
        # Queremos saber "en qué se fijó el modelo para deducir esa edad"
        age_pred.backward(retain_graph=True)
        
        # Global Average Pooling de los gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Ponderar las activaciones con la importancia de los gradientes deducida antes
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Sumar los canales para crear el heatmap final y aplicar ReLU (ignorar lo negativo)
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalizar entre 0 y 1
        heatmap_max = np.max(heatmap)
        if heatmap_max == 0:
            return original_image_np
            
        heatmap /= heatmap_max
        
        # Redimensionar el heatmap para que coincida con la foto subida
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Darle a las temperaturas la paleta JET (zonas calientes en rojo)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Superponer (40% de color y 60% la foto original)
        superimposed_img = heatmap_color * 0.4 + original_image_np * 0.6
        return np.uint8(superimposed_img), class_prob, age_pred
