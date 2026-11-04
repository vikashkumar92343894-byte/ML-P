import os
import time
import imageio.v3 as iio
from PIL import Image, ImageDraw
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

TARGET_SIZE = (1280, 720)  # Standard 720p HD (divisible by 16)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/frames", exist_ok=True)

def create_title_slide(title: str, subtitle: str, size=TARGET_SIZE, duration_frames=20):
    img = Image.new("RGB", size, color="#0F172A")
    draw = ImageDraw.Draw(img)
    
    # Top and bottom accent lines
    draw.rectangle([0, 0, size[0], 6], fill="#3B82F6")
    draw.rectangle([0, size[1]-6, size[0], size[1]], fill="#3B82F6")
    
    # Text
    draw.text((80, 240), title, fill="#F8FAFC")
    draw.text((80, 310), subtitle, fill="#94A3B8")
    draw.text(
        (80, 390),
        "Key Capabilities Showcase:\n"
        "• Automatic Recurring Detection (Random Forest ML - F1: 0.9988)\n"
        "• Silent Price Hike Alerting (> 3% increase)\n"
        "• Accidental Duplicate Charge Detection (< 24 hours)\n"
        "• Unsupervised Outlier Anomaly Detection (Isolation Forest)\n"
        "• High-Performance FastAPI Microservice (< 50ms latency)\n"
        "• Interactive Streamlit Analytics Dashboard",
        fill="#38BDF8"
    )
    
    return [img] * duration_frames

def capture_demo_frames():
    print("Setting up automated Chrome session...")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={TARGET_SIZE[0]},{TARGET_SIZE[1]}")
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    
    driver = webdriver.Chrome(options=options)
    raw_frames = []
    
    try:
        # 1. Intro Slide
        print("1. Generating Intro Title Slide...")
        raw_frames.extend(create_title_slide(
            "Hidden Subscription & Expense Detector",
            "Production-Grade ML Expense Intelligence System Demo",
            duration_frames=20
        ))
        
        # 2. Streamlit Dashboard Overview
        print("2. Loading Streamlit Dashboard (http://localhost:8501)...")
        driver.get("http://localhost:8501")
        time.sleep(5)  # Wait for Streamlit connection & state
        
        frame1_path = "reports/frames/01_dashboard_overview.png"
        driver.save_screenshot(frame1_path)
        raw_frames.extend([Image.open(frame1_path)] * 25)
        
        # 3. Scroll to Subscriptions Table
        print("3. Capturing Subscriptions Table...")
        driver.execute_script("window.scrollBy(0, 220);")
        time.sleep(1)
        frame2_path = "reports/frames/02_subscriptions_table.png"
        driver.save_screenshot(frame2_path)
        raw_frames.extend([Image.open(frame2_path)] * 25)
        
        # 4. Tab 2: Risk & Alerts Center
        print("4. Switching to Tab 2: Risk & Alerts Center...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[data-baseweb='tab']")
        if len(tabs) >= 2:
            tabs[1].click()
            time.sleep(2)
            frame3_path = "reports/frames/03_alerts_center.png"
            driver.save_screenshot(frame3_path)
            raw_frames.extend([Image.open(frame3_path)] * 30)
            
        # 5. Tab 3: Spending Analytics
        print("5. Switching to Tab 3: Spending Analytics...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[data-baseweb='tab']")
        if len(tabs) >= 3:
            tabs[2].click()
            time.sleep(3)
            frame4_path = "reports/frames/04_spending_analytics.png"
            driver.save_screenshot(frame4_path)
            raw_frames.extend([Image.open(frame4_path)] * 30)
            
        # 6. Tab 4: Live Transaction Tester
        print("6. Switching to Tab 4: Live Transaction Tester...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[data-baseweb='tab']")
        if len(tabs) >= 4:
            tabs[3].click()
            time.sleep(1)
            submit_btn = driver.find_elements(By.CSS_SELECTOR, "button[kind='secondaryFormSubmit']")
            if submit_btn:
                submit_btn[0].click()
                time.sleep(2)
            frame5_path = "reports/frames/05_live_tester.png"
            driver.save_screenshot(frame5_path)
            raw_frames.extend([Image.open(frame5_path)] * 30)

        # 7. FastAPI Swagger Docs
        print("7. Navigating to FastAPI Swagger Docs (http://localhost:8000/docs)...")
        driver.get("http://localhost:8000/docs")
        time.sleep(3)
        frame6_path = "reports/frames/06_fastapi_swagger.png"
        driver.save_screenshot(frame6_path)
        raw_frames.extend([Image.open(frame6_path)] * 25)
        
    finally:
        driver.quit()
        
    print(f"Total raw frames captured: {len(raw_frames)}")
    
    # Standardize all frames to exact TARGET_SIZE (1280, 720)
    print(f"Resizing and normalizing all frames to {TARGET_SIZE}...")
    standardized_frames = [
        im.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        for im in raw_frames
    ]
    
    # 1. Compile MP4 Video
    video_path = "reports/figures/demo_video.mp4"
    print(f"Rendering high-definition MP4 video to {video_path}...")
    import numpy as np
    frame_arrays = [np.array(im) for im in standardized_frames]
    iio.imwrite(video_path, frame_arrays, fps=10, codec="libx264")
    print(f"Video saved! Size: {os.path.getsize(video_path):,} bytes.")

    # 2. Compile Animated GIF for GitHub README
    gif_path = "reports/figures/demo.gif"
    print(f"Rendering animated GIF to {gif_path} for GitHub README...")
    # Downscale GIF slightly to 960x540 for fast loading on GitHub (< 5MB)
    gif_frames = [
        im.resize((960, 540), Image.Resampling.LANCZOS)
        for im in standardized_frames[::2]  # Subsample every 2nd frame
    ]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
        optimize=True
    )
    print(f"GIF saved! Size: {os.path.getsize(gif_path):,} bytes.")
    
    return video_path, gif_path

if __name__ == "__main__":
    capture_demo_frames()
