@echo off
echo.
echo ========================================
echo    AIXHost.exe Troubleshooting
echo ========================================
echo.

echo Checking for AIXHost.exe processes...
tasklist | findstr /i "AIXHost"

echo.
echo Checking AI-related processes...
tasklist | findstr /i "AI"

echo.
echo If AIXHost.exe is running, you can try:
echo 1. Close any AI software (voice assistants, AI tools)
echo 2. End the process: taskkill /f /im AIXHost.exe
echo 3. Restart your computer
echo 4. Check Windows Task Scheduler for AI-related tasks
echo.

echo Press any key to check system processes...
pause >nul

echo.
echo Full process list (looking for AI-related):
tasklist /fo table | findstr /i "AI"

echo.
echo Checking Windows services:
sc query | findstr /i "AI"

echo.
echo Done. If you see AIXHost, it's likely from other AI software.
pause