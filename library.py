from angle_emb import AnglE, Prompts
from chromadb.utils import embedding_functions
import chromadb
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pytesseract
import cv2
from pytesseract import Output
import os
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import argparse
