import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from torch.autograd import Variable
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from u2net import U2NET

class BackgroundRemover:
    def __init__(self, model_path='saved_models/u2net.pth'):
        """
        Initialize U2-Net model for background removal
        """
        self.model_path = model_path
        self.net = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load U2-Net model"""
        print("Loading U2NET model...")
        self.net = U2NET(3, 1)
        
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(self.model_path, weights_only=False))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=False))
        
        self.net.eval()
        print("U2NET model loaded successfully")
    
    def _normPRED(self, d):
        """Normalize prediction"""
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d-mi)/(ma-mi)
        return dn
    
    def _preprocess_image(self, image):
        """Preprocess image for U2-Net"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Create sample dict for data loader
        sample = {
            'imidx': np.array([0]),
            'image': image_np,
            'label': np.zeros_like(image_np[:,:,0:1])  # dummy label
        }
        
        # Apply transforms
        transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
        sample = transform(sample)
        
        # Add batch dimension
        inputs = sample['image'].unsqueeze(0).type(torch.FloatTensor)
        
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
            
        return inputs
    
    def remove_background_from_url(self, image_url):
        """
        Remove background from image URL and return PIL Image with white background
        
        Args:
            image_url (str): URL of the image
            
        Returns:
            PIL.Image: Image with white background
        """
        try:
            # Download image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            return self.remove_background_from_image(image)
            
        except Exception as e:
            print(f"Error processing image from URL {image_url}: {str(e)}")
            return None
    
    def remove_background_from_image(self, image):
        """
        Remove background from PIL Image and return image with white background
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with white background
        """
        try:
            original_size = image.size
            
            # Preprocess image
            inputs = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = self.net(inputs)
            
            # Get prediction and normalize
            pred = d1[:, 0, :, :]
            pred = self._normPRED(pred)
            
            # Convert to numpy
            predict = pred.squeeze().cpu().data.numpy()
            mask = (predict * 255).astype(np.uint8)
            
            # Convert original image to RGB numpy array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image)
            
            # Resize mask to match original image size
            mask_pil = Image.fromarray(mask).resize(original_size, resample=Image.BILINEAR)
            mask_np = np.array(mask_pil) / 255.0  # Normalize to 0-1
            
            # Create white background
            white_bg = np.ones_like(image_np, dtype=np.uint8) * 255
            
            # Apply mask (foreground keeps original, background becomes white)
            result = image_np * mask_np[..., None] + white_bg * (1 - mask_np[..., None])
            result = result.astype(np.uint8)
            
            # Convert back to PIL Image
            result_pil = Image.fromarray(result)
            
            # Clean up GPU memory
            del d1, d2, d3, d4, d5, d6, d7, inputs, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result_pil
            
        except Exception as e:
            print(f"Error removing background: {str(e)}")
            return None

def remove_background_batch(product_datas):
    """
    Remove background from batch of product images
    
    Args:
        product_datas: List of tuples [(product_id, image_url), ...]
        
    Returns:
        List of tuples [(product_id, processed_image), ...]
        processed_image is PIL.Image with white background
    """
    remover = BackgroundRemover()
    results = []
    
    for product_id, image_url in product_datas:
        print(f"Processing product {product_id}...")
        processed_image = remover.remove_background_from_url(image_url)
        
        if processed_image is not None:
            results.append((product_id, processed_image))
        else:
            print(f"Failed to process product {product_id}")
    
    return results

# For standalone testing
def main():
    """Test function"""
    # Test with sample data
    test_image_path = "test_data/test_images"
    if os.path.exists(test_image_path):
        import glob
        
        remover = BackgroundRemover()
        image_files = glob.glob(os.path.join(test_image_path, "*"))
        
        for img_path in image_files[:1]:  # Test with first image
            print(f"Testing with {img_path}")
            image = Image.open(img_path)
            result = remover.remove_background_from_image(image)
            
            if result:
                output_path = f"test_output_{os.path.basename(img_path)}"
                result.save(output_path)
                print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main() 