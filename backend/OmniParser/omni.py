import os
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from PIL import Image
import io
import base64
import torch
import pandas as pd
import uvicorn

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Also log to file for persistent debugging
file_handler = logging.FileHandler('omni_parser.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# 初始化 FastAPI
logger.info("Initializing FastAPI application")
app = FastAPI()

# 默认设备
logger.info("Detecting available device")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# 初始化模型，只加载一次
logger.info("Starting model initialization")
yolo_model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
caption_model_name = 'florence2'
caption_model_path = 'weights/icon_caption_florence'

try:
    logger.info(f"Loading YOLO model from: {yolo_model_path}")
    som_model = get_yolo_model(model_path=yolo_model_path)
    logger.info("YOLO model loaded successfully")
    
    logger.info(f"Moving YOLO model to device: {device}")
    som_model.to(device)
    logger.info("YOLO model moved to device successfully")
    
    logger.info(f"Loading caption model: {caption_model_name} from: {caption_model_path}")
    caption_model_processor = get_caption_model_processor(
        model_name=caption_model_name,
        model_name_or_path=caption_model_path,
        device=device
    )
    logger.info("Caption model loaded successfully")
    logger.info("All models initialized successfully")
    
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    logger.exception("Full model initialization error traceback:")
    raise

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = Form(0.05), # Box Threshold
    iou_threshold: float = Form(0.1),  # IOU Threshold
    imgsz_component: int = Form(640)  # Icon Detect Image Size
):
    logger.info(f"New image processing request received")
    logger.info(f"Parameters - box_threshold: {box_threshold}, iou_threshold: {iou_threshold}, imgsz_component: {imgsz_component}")
    logger.info(f"Uploaded file - name: {file.filename}, content_type: {file.content_type}")
    
    try:
        # 保存上传文件到临时路径
        logger.debug("Reading uploaded file contents")
        contents = await file.read()
        logger.info(f"File contents read successfully, size: {len(contents)} bytes")
        
        logger.debug("Creating temporary directory")
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, file.filename)
        logger.info(f"Temporary file path: {temp_image_path}")

        logger.debug("Writing file to temporary location")
        with open(temp_image_path, 'wb') as f:
            f.write(contents)
        logger.info("File written to temporary location successfully")
        
        logger.debug("Opening and converting image to RGB")
        image = Image.open(temp_image_path).convert('RGB')
        image_size = image.size
        logger.info(f"Image opened successfully, size: {image_size}")
        
        start_time = time.time()
        logger.info("Starting image processing pipeline")
        
        # OCR 检测
        logger.info("Starting OCR detection")
        try:
            ocr_bbox_rslt, _ = check_ocr_box(
                temp_image_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.5},
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR detection completed successfully, found {len(ocr_bbox) if ocr_bbox else 0} text boxes")
            logger.debug(f"OCR text results: {text}")
        except Exception as e:
            logger.error(f"Error during OCR detection: {str(e)}")
            logger.exception("OCR detection error traceback:")
            raise
        
        # 生成标注图像和解析内容
        logger.info("Configuring bounding box drawing parameters")
        draw_bbox_config = {
            'text_scale': 0.8 * (max(image.size) / 3200),
            'text_thickness': max(int(2 * (max(image.size) / 3200)), 1),
            'text_padding': max(int(3 * (max(image.size) / 3200)), 1),
            'thickness': max(int(3 * (max(image.size) / 3200)), 1),
        }
        logger.debug(f"Draw bbox config: {draw_bbox_config}")
        
        logger.info("Starting SOM (Set of Mark) labeling and processing")
        try:
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                temp_image_path,
                som_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=iou_threshold,
                scale_img=False,
                batch_size=128,
                imgsz=imgsz_component
            )
            elapsed_time = time.time() - start_time
            logger.info(f"SOM labeling completed successfully in {elapsed_time:.2f} seconds")
            logger.info(f"Generated {len(parsed_content_list) if parsed_content_list else 0} parsed content items")
            logger.info(f"Generated {len(label_coordinates) if label_coordinates else 0} label coordinates")
        except Exception as e:
            logger.error(f"Error during SOM labeling: {str(e)}")
            logger.exception("SOM labeling error traceback:")
            raise
        # 删除临时文件
        logger.debug("Cleaning up temporary files")
        try:
            os.remove(temp_image_path)
            os.rmdir(temp_dir)
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

        # 返回标注图片和解析内容
        logger.info("Processing results for response")
        try:
            logger.debug("Decoding labeled image from base64")
            image_bytes = base64.b64decode(dino_labeled_img)
            labeled_image = io.BytesIO(image_bytes)
            logger.debug("Labeled image decoded successfully")

            # 解析内容转 DataFrame -> JSON
            logger.debug("Converting parsed content to DataFrame")
            df = pd.DataFrame(parsed_content_list)
            df['ID'] = range(len(df))
            parsed_content_json = df.to_dict(orient="records")
            logger.info(f"Created DataFrame with {len(df)} rows")

            # base64 编码
            logger.debug("Encoding final image to base64")
            encoded_image = base64.b64encode(labeled_image.getvalue())
            logger.info("Final image encoded successfully")
            
            logger.info(f"Request completed successfully in {elapsed_time:.2f} seconds")
            return {
                "status": "success",
                "parsed_content": parsed_content_json,
                "labeled_image": encoded_image,
                "e_time": elapsed_time  # 返回耗时
            }
        except Exception as e:
            logger.error(f"Error processing results for response: {str(e)}")
            logger.exception("Result processing error traceback:")
            raise

    except Exception as e:
        logger.error(f"Unhandled error in process_image: {str(e)}")
        logger.exception("Full error traceback:")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    logger.info("Starting OmniParser API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# nohup fastapi run omni.py --port 8000 > ../logfile_omni.log 2>&1