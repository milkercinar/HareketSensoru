import cv2
import time
import pyautogui
import webbrowser
import os

def send_message_whatsapp():
    phone_number = "+905414337271"
    timestamp = time.strftime("%d %H:%M")
    message = f"Hareket tespit edildi! Tarih/Saat:{timestamp}"
    web_url = f"https://web.whatsapp.com/send?phone={phone_number}&text={message}"

    try:
        print(f"[INFO] Whatsapp üzerinden mesaj gönderiliyor: {message}")
        webbrowser.open(web_url)
        time.sleep(20)
        pyautogui.press('enter')
        time.sleep(5)
        pyautogui.hotkey('ctrl','w')
        time.sleep(2)
        print("[INFO] Whatsapp mesajı gönderildi!")

    except Exception as e:
        print("[ERROR] Whatsapp gönderiminde hata oluştu", e)
        return False   


def main():
    pyautogui.PAUSE = 1.5

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera Açılamadı")
        return
    
    backSub = cv2.createBackgroundSubtractorMOG2(
        history = 500,
        varThreshold = 25,
        detectShadows = True
    )

    last_motion_time = 0
    cooldown = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Kamera görüntüsü alınamadı!")
            break

        fgMask = backSub.apply(frame)
        _ ,thresh = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                continue
            motion_detected = True
            break
        current_time = time.time()

        if motion_detected and (current_time - last_motion_time > cooldown):
            print("[INFO] Hareket Algılandı!")

            if send_message_whatsapp():
                last_motion_time = current_time
                cv2.putText(frame, "Hareket algılandı!")
            else:
                print("[WARNING] Whatsapp mesajı gönderilemedi!")

        else:
            cv2.putText(frame, "Hareket algılandı!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Kamera", frame)
        cv2.imshow("Hareket Maskesi", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cap.destroyAllWindows()

if __name__ == "__main__":
    main()                       



