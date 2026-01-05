import torch #$goi thu vien pytorch
from diffusers import StableDiffusionPipeline
import os

def setup_ai_painter():
    print("--- Đang khởi tạo AI Họa Sĩ (Stable Diffusion) ---")
    print("Lưu ý: Lần đầu chạy sẽ cần tải mô hình (~4GB) từ Internet...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên thiết bị: {device.upper()}")

    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        if device == "cuda":
            #dùng float16
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            #dùng CPU
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            
        pipe = pipe.to(device)
        # Tắt bộ lọc an toàn
        pipe.safety_checker = None 
        print("--- AI đã sẵn sàng! ---")
        return pipe
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Hãy đảm bảo bạn đã cài đủ thư viện: pip install diffusers transformers accelerate")
        return None

def paint(pipe, prompt, filename="output.png"):
    print(f"\nĐang vẽ: '{prompt}' ... (Vui lòng đợi)")
    
    # Thực hiện vẽ
    image = pipe(prompt, num_inference_steps=30).images[0]
    
    # Lưu ảnh
    image.save(filename)
    print(f"Đã vẽ xong! Ảnh được lưu tại: {os.path.abspath(filename)}")
    
    # Hiển thị ảnh
    os.startfile(filename)

if __name__ == "__main__":
    ai_pipe = setup_ai_painter()
    
    if ai_pipe:
        while True:
            user_prompt = input("\nNhập mô tả tranh bạn muốn vẽ (Tiếng Anh, gõ 'exit' để thoát): ")
            if user_prompt.lower() == 'exit':
                break
            
            # Tạo tên file
            import time
            file_name = f"ai_art_{int(time.time())}.png"
            
            paint(ai_pipe, user_prompt, file_name)