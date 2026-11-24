@echo off
chcp 65001 >nul

echo ========================================================
echo   KHỞI CHẠY TRỢ LÝ AI LOCAL CÁ NHÂN (OFFLINE 100%%)
echo ========================================================

where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [!] Ollama chưa được cài đặt trong hệ thống.
    echo [i] Vui lòng tải và cài đặt tại: https://ollama.com/download
    pause
    exit /b 1
)

echo [*] Đang nạp cấu hình và tạo mô hình 'my-ai' từ Modelfile...
ollama create my-ai -f Modelfile

if %ERRORLEVEL% EQU 0 (
    echo [+] Tạo mô hình 'my-ai' thành công!
    echo [*] Đang khởi động phiên trò chuyện trực tiếp...
    ollama run my-ai
) else (
    echo [!] Gặp lỗi khi tạo mô hình. Vui lòng kiểm tra lại Modelfile hoặc tệp GGUF.
)

pause
