import os
import logging
from fastapi import APIRouter, File, UploadFile, Form
from insert_clipper import process_dxf_bytes, process_dxf_file
from threading import Thread
from uuid import uuid4
import traceback
from io import BytesIO
from fastapi import HTTPException

router = APIRouter()
task_status = {}


def setup_logger(task_id, log_folder):
    logger = logging.getLogger(task_id)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_folder, f"task_{task_id}.log"))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def create_logging_path(task_id):
    TASK_DIR = f"/tmp/tasks_log/{task_id}"
    os.makedirs(TASK_DIR, exist_ok=True)
    
    return TASK_DIR

def process_cad(task_id, content):
    task_status[task_id] = {"status" : "running", "exclude_ids" : None}
    TASK_DIR = create_logging_path(task_id)
    
    filename = os.path.join(TASK_DIR, f"{task_id}.dxf")

    logger = setup_logger(task_id, TASK_DIR)
    logger.info(f"Task {task_id} created")

    try:
        logger.info("Starting task")
        
        with open(filename, "wb") as f:
            f.write(content)

        exclude_ids = process_dxf_file(filename)

        task_status[task_id]["status"] = "done"
        task_status[task_id]["exclude_ids"] = exclude_ids

    except Exception as e:
        error_msg = f"Task {task_id} failed with error {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        task_status[task_id] = {"status": "error", "exclude_ids": None}

@router.post("/clip-ids/start-task/")
async def process_svg(file: UploadFile = File(...)):
    task_id = str(uuid4())
    content = await file.read()

    thread = Thread(target=process_cad,args=(task_id, content))
    thread.start()
    
    return { "task_id": task_id }

@router.get("/clip-ids/task/{task_id}")
def get_status(task_id: str):
    status = task_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return status