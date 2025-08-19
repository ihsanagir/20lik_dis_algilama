import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

print("YOLO modeli yükleniyor...")
model = YOLO("trained_models/disprojesi2"
             "/weights/best.pt")
print("Model başarıyla yüklendi.")

def detect_wisdom_teeth(input_image, confidence_threshold):
    if input_image is None:
        return None, "Lütfen bir resim yükleyin."

    results = model(input_image, conf=confidence_threshold)

    annotated_image = input_image.copy()

    sayac = 1
    detection_summary = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"{sayac}.Dis"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            detection_summary.append(f"{sayac}. Diş bulundu. Güven Skoru: {conf:.2%}")
            sayac += 1

    if not detection_summary:
        summary_text = "Belirtilen güven aralığında herhangi bir yirmilik diş tespit edilemedi."
    else:
        summary_text = "\n".join(detection_summary)

    return annotated_image, summary_text

iface = gr.Interface(
    fn=detect_wisdom_teeth,
    inputs=[
        gr.Image(type="numpy",
                 label="İncelenecek Röntgen Fotoğrafını Yükleyin",
                 sources=["upload"],
                 width=640,
                 height=480),
        gr.Slider(
            minimum=0.05,
            maximum=1.00,
            step=0.05,
            value=0.30,  # Varsayılan değer %30
            label="Güven Eşiği",
            info="Modelin ne kadar doğruluğundan emin olması gerektiğini seçin.\nDüşük değerler daha fazla ama potansiyel olarak yanlış diş bulabilir."
        )
    ],
    outputs=[
        gr.Image(type="numpy",
                 label="Tespit Sonucu",
                 width=640,
                 height=480),
        gr.Textbox(label="Tespit Özeti")
    ],
    title="🦷 Yirmilik Diş Tespit Uygulaması",
    description="Bir fotoğraf yükleyin, güven eşiğini ayarlayın ve sonucu görün.",
    flagging_mode="never",
submit_btn = "Gönder",
clear_btn = "Temizle"
)
if __name__ == "__main__":
    iface.launch(share=True)