import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional
import math
import time
import fitz  # PyMuPDF
from pydub import AudioSegment
from indextts.infer import IndexTTS
import numpy as np
from scipy.io import wavfile
import noisereduce as nr  # 新增降噪库

def log_message(message: str, level: str = "INFO") -> None:
    """记录带有时间戳和级别的消息"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")

def check_path(path: str, is_file: bool = True) -> None:
    """检查路径是否存在；如果不存在则抛出错误"""
    log_message(f"检查路径: {path}")
    path_obj = Path(path)
    if is_file and not path_obj.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not is_file and not path_obj.is_dir():
        raise NotADirectoryError(f"目录不存在: {path}")

def natural_sort_key(s: str) -> list:
    """用于文件名自然排序的键函数"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def get_file_size_mb(file_path: str) -> float:
    """获取文件大小（MB）"""
    size = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
    log_message(f"文件大小: {file_path} - {size:.2f} MB")
    return size

def estimate_batch_size(part_files: List[str], input_dir: str, max_mb: float = 3800.0) -> int:
    """根据平均文件大小估计批处理大小"""
    if not part_files:
        return 0
    sample_size = min(10, len(part_files))
    avg_size = sum(get_file_size_mb(os.path.join(input_dir, f)) for f in part_files[:sample_size]) / sample_size
    batch_size = max(1, int(max_mb / avg_size)) if avg_size > 0 else 1
    log_message(f"平均文件大小: {avg_size:.2f} MB, 批处理大小: {batch_size}")
    return batch_size

def merge_batch(batch_files: List[str], batch_num: int, input_dir: str, output_dir: str) -> str:
    """将一批音频文件合并为单个WAV文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"batch_{batch_num}.wav")
    combined = AudioSegment.empty()
    for file in batch_files:
        combined += AudioSegment.from_wav(os.path.join(input_dir, file))
    combined.export(output_file, format="wav")
    log_message(f"合并批处理 {batch_num}: {output_file}")
    return output_file

def batch_merge_wav(input_dir: str, output_dir: str, max_mb: float = 3800.0) -> List[str]:
    """批量合并WAV文件"""
    part_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav') and '.part' in f], key=natural_sort_key)
    if not part_files:
        log_message("未找到WAV部分文件", "ERROR")
        return []
    
    files_per_batch = estimate_batch_size(part_files, input_dir, max_mb)
    total_batches = math.ceil(len(part_files) / files_per_batch)
    output_files = [
        merge_batch(part_files[i*files_per_batch:(i+1)*files_per_batch], i+1, input_dir, output_dir)
        for i in range(total_batches)
    ]
    log_message(f"批量合并完成，生成 {len(output_files)} 个文件")
    return output_files

def split_text(text: str, max_length: int = 10000) -> List[str]:
    """将文本分割成块"""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def preprocess_audio(input_path: str, duration_sec: float = 8.0) -> str:
    """通过裁剪到指定时长来预处理音频提示，并保存在当前目录"""
    check_path(input_path)
    audio = AudioSegment.from_file(input_path)
    segment = audio[:int(duration_sec * 1000)]
    
    # 获取输入文件的基本名称
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # 构建输出文件名，添加预处理标记
    output_file = os.path.join(os.getcwd(), f"{base_name}_processed.wav")
    
    segment.export(output_file, format="wav")
    log_message(f"预处理音频: {output_file}")
    return output_file

def reduce_noise(input_path: str, output_path: str, noise_level: float = 0.5) -> bool:
    """对音频文件进行降噪处理
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出降噪后音频文件路径
        noise_level: 降噪强度，范围0-1，值越大降噪越强烈
        
    Returns:
        处理是否成功
    """
    try:
        log_message(f"对 {input_path} 进行降噪处理")
        # 读取音频文件
        rate, data = wavfile.read(input_path)
        
        # 确保音频数据是numpy数组
        if isinstance(data, np.ndarray):
            # 对单声道或立体声进行处理
            if len(data.shape) == 1:  # 单声道
                reduced_noise = nr.reduce_noise(
                    y=data, 
                    sr=rate, 
                    stationary=False,
                    prop_decrease=noise_level,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512
                )
            else:  # 立体声
                reduced_noise = np.zeros_like(data)
                for ch in range(data.shape[1]):
                    reduced_noise[:, ch] = nr.reduce_noise(
                        y=data[:, ch], 
                        sr=rate, 
                        stationary=False,
                        prop_decrease=noise_level,
                        n_fft=2048,
                        win_length=2048,
                        hop_length=512
                    )
            
            # 保存降噪后的音频
            wavfile.write(output_path, rate, reduced_noise)
            log_message(f"降噪完成，已保存至 {output_path}")
            return True
        else:
            log_message("音频数据格式不正确，无法进行降噪处理", "ERROR")
            return False
    except Exception as e:
        log_message(f"降噪处理失败: {e}", "ERROR")
        return False

def extract_pdf_text(pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """从指定页面范围内的PDF中提取文本"""
    check_path(pdf_path)
    with fitz.open(pdf_path) as pdf:
        end_page = pdf.page_count - 1 if end_page is None else min(end_page, pdf.page_count - 1)
        if start_page < 0 or start_page > end_page:
            log_message("页面范围无效", "WARNING")
            return ""
        text = "".join(pdf[page].get_text() + "\n" for page in range(start_page, end_page + 1))
    log_message(f"从PDF中提取了 {len(text)} 个字符")
    return text

def read_txt_file(txt_path: str) -> str:
    """使用编码回退读取TXT文件文本"""
    check_path(txt_path)
    for encoding in ('utf-8', 'gbk'):
        try:
            with open(txt_path, 'r', encoding=encoding) as file:
                text = file.read()
                log_message(f"从TXT中读取了 {len(text)} 个字符")
                return text
        except UnicodeDecodeError:
            log_message(f"使用 {encoding} 编码读取失败", "WARNING")
    log_message("使用支持的编码读取TXT文件失败", "ERROR")
    return ""

def text_to_speech(text: str, audio_prompt_path: str, output_path: str, 
                  audio_duration: float = 8.0, tts: Optional[IndexTTS] = None,
                  apply_denoise: bool = True, noise_level: float = 0.5) -> List[str]:
    """使用IndexTTS将文本转换为语音
    
    Args:
        text: 要转换的文本
        audio_prompt_path: 音频提示文件路径
        output_path: 输出音频文件路径
        audio_duration: 音频提示持续时间（秒）
        tts: IndexTTS实例
        apply_denoise: 是否应用降噪处理
        noise_level: 降噪强度，范围0-1
        
    Returns:
        生成的音频文件列表
    """
    tts = tts or IndexTTS(
        cfg_path="/app/sda1/xiangyue/model/IndexTTS-1.5/config.yaml",
        model_dir="/app/sda1/xiangyue/model/IndexTTS-1.5",
        is_fp16=True,
        use_cuda_kernel=False
    )
    prompt_wav = preprocess_audio(audio_prompt_path, audio_duration)
    text_chunks = split_text(text)
    output_files = []
    
    for i, chunk in enumerate(text_chunks):
        temp_output_file = f"{output_path}.part{i}.wav"
        success = tts.infer_fast(
            audio_prompt=prompt_wav,
            text=chunk,
            output_path=temp_output_file,
            max_text_tokens_per_sentence=120,
            sentences_bucket_max_size=2,
            max_mel_tokens=1024
        )
        
        if success:
            if apply_denoise:
                # 对生成的音频进行降噪处理
                denoised_output_file = f"{output_path}.part{i}.denoised.wav"
                if reduce_noise(temp_output_file, denoised_output_file, noise_level):
                    # 降噪成功，删除原始文件并使用降噪后的文件
                    os.remove(temp_output_file)
                    os.rename(denoised_output_file, temp_output_file)
                    log_message(f"已对 {temp_output_file} 进行降噪处理")
                else:
                    log_message(f"对 {temp_output_file} 降噪失败，使用原始文件", "WARNING")
            
            output_files.append(temp_output_file)
        else:
            log_message(f"处理文本块 {i+1} 失败", "ERROR")
    
    log_message(f"文本到语音转换完成，生成 {len(output_files)} 个文件")
    return output_files

def delete_files(file_list: List[str]) -> None:
    """删除指定文件"""
    for file in file_list:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                log_message(f"删除 {file} 失败: {e}", "WARNING")

def merge_audio_files(input_files: List[str], output_path: str, max_mb: float = 3800.0) -> List[str]:
    """将音频文件合并为单个文件"""
    if not input_files:
        log_message("没有要合并的输入文件", "WARNING")
        return []
    if len(input_files) == 1:
        log_message("单个输入文件，无需合并")
        return input_files
    
    input_dir = os.path.dirname(input_files[0])
    output_dir = os.path.dirname(output_path)  # 使用输出路径的目录
    original_files = input_files.copy()
    
    batch_outputs = batch_merge_wav(input_dir, output_dir, max_mb)
    if len(batch_outputs) == 1:
        os.rename(batch_outputs[0], output_path)
        batch_outputs = [output_path]
    
    delete_files(original_files)
    log_message(f"音频合并完成，生成 {len(batch_outputs)} 个文件")
    return batch_outputs

def process_document(
    input_path: str,
    audio_prompt_path: str,
    output_path: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
    audio_duration: float = 8.0,
    tts: Optional[IndexTTS] = None,
    max_batch_size_mb: float = 3800.0,
    apply_denoise: bool = True,
    noise_level: float = 0.5
) -> List[str]:
    """处理文档（PDF/TXT）转换为音频
    
    Args:
        input_path: 输入文档路径
        audio_prompt_path: 音频提示文件路径
        output_path: 输出音频文件路径
        start_page: PDF起始页（从0开始）
        end_page: PDF结束页
        audio_duration: 音频提示持续时间（秒）
        tts: IndexTTS实例
        max_batch_size_mb: 最大批处理大小（MB）
        apply_denoise: 是否应用降噪处理
        noise_level: 降噪强度，范围0-1
        
    Returns:
        生成的音频文件列表
    """
    log_message(f"处理文档: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    text = extract_pdf_text(input_path, start_page, end_page) if input_path.endswith('.pdf') else read_txt_file(input_path)
    if not text:
        log_message("未提取到文本", "ERROR")
        return []
    
    audio_chunks = text_to_speech(
        text, 
        audio_prompt_path, 
        output_path, 
        audio_duration, 
        tts,
        apply_denoise,
        noise_level
    )
    if not audio_chunks:
        log_message("未生成音频块", "WARNING")
        return []
    
    return merge_audio_files(audio_chunks, output_path, max_batch_size_mb)

if __name__ == "__main__":
    try:
        tts = IndexTTS(
            cfg_path="/app/sda1/xiangyue/model/IndexTTS-1.5/config.yaml",
            model_dir="/app/sda1/xiangyue/model/IndexTTS-1.5",
            is_fp16=True,
            use_cuda_kernel=False
        )
        input_path = "测试频段.txt"
        audio_prompt_path = "配音木城.mp3"
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}/{base_name}.wav"
        
        output_files = process_document(
            input_path=input_path,
            audio_prompt_path=audio_prompt_path,
            output_path=output_path,
            tts=tts,
            apply_denoise=False,
            audio_duration=40   
        )
        
        if output_files:
            print("输出文件:")
            for i, file in enumerate(output_files):
                print(f"{i+1}. {file} - {get_file_size_mb(file):.2f} MB")
        else:
            print("处理失败: 未生成输出文件")
    except Exception as e:
        log_message(f"错误: {e}", "ERROR")
        print(f"错误: {e}")
