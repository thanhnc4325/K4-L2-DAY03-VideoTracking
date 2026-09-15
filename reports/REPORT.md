# Báo cáo Ngày 3 — Tracking Annotation

Họ tên: Nguyễn Công Thành
Ngày: 15/09/2026

---

## 1. Quá trình gán nhãn

| Mục | Giá trị |
| --- | --- |
| Công cụ | CVAT |
| Thời gian gán clip_02 (warm-up) | 36 phút |
| Thời gian gán clip_01 | 42 phút |
| Số track đã vẽ trong clip_01 | 8 |
| Số keyframe trung bình mỗi track | `...` |

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. Một số xe đè sát lên nhau, và sử dụng phóng to để căn chỉnh chính xác nhất
2. Một số hình ảnh chuyển động, và sử dụng phóng to cùng với nhiều thời gian để đưa ra khung chính xác nhất
3. Có thời điểm xuất hiện nhiều xe một lúc, và check thật kỹ để không bỏ sót bất kỳ vật thể hay chi tiết nào

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: Lượt một phát hiện bỏ sót mất một đoạn xe buýt mới vào khung hình do phóng to khung hình 
- Lượt 2: Mọi thứ đều ổ, chỉ cần chỉnh sửa một chút các khung hình cho mượn và chính xác nhất
- Lượt 3: Mọi thứ đã khá chuẩn xác, không còn sai sót nào xảy ra

Kiểm chéo với: `...`. Chi tiết ở `reports/review_partner.md`.
Số lỗi bạn tìm được trong bản của bạn ấy: `...`. Số lỗi bạn ấy tìm được trong bản của bạn: `...`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`...`

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence | Giá trị |
| --- | --- |
| SHA-256 từ evidence/pre-gold/clip_01/manifest.json | 90ebbe59734750b21e2286973521e9025c431f8dc5ea44d1b592792b208b63d0 |
| Thời điểm khóa | 651 |
| Số row / frame / track trước khi mở reference | 8 |

| | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bản pre-gold | 0.768 | 0.734 | 0.811 | 0.866 | 0.936 | 0.864 | 0.847 | 78 | 0 | 0 |
| Sau rework | 0.845 | 0.890 | 0.815 | 0.895 | 0.940 | 0.985 | 0.860 | 5 | 0 | 0 |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): Có

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
| --- | --- | --- | --- |
| Bbox treo | 51-78 | 5 | Chỉnh lại frame rời khung thành outside |
| Bbox treo | 73-100 | 6 | Đánh outside đúng frame xe đã ra mép |
| Bbox trôi | 101-104 | 6 | Thêm keyframe quanh các frame bị lỏng IoU |
| Bbox trôi | 92 | 5 | Thêm keyframe để khoanh sát mép xe hơn |

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục | Giá trị |
| --- | --- |
| Python / ultralytics / torch / lap | 3.13.15 / 8.4.145 / 2.11.0+cu128 / 0.5.13 |
| weights / hai tracker | yolo26n.pt / bytetrack.yaml, botsort-reid.yaml |
| conf / IoU / imgsz / classes | 0.25 / 0.7 / 960 / [2, 5, 7] |
| device | 0 (GPU) |

| So sánh | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bạn vs gold | 0.768 | 0.734 | 0.811 | 0.866 | 0.936 | 0.864 | 0.847 | 78 | 0 | 0 |
| ByteTrack control vs gold | 0.709 | 0.649 | 0.776 | 0.846 | 0.875 | 0.749 | 0.823 | 88 | 54 | 2 |
| BoT-SORT + ReID vs gold | 0.763 | 0.711 | 0.820 | 0.872 | 0.900 | 0.792 | 0.860 | 91 | 26 | 2 |
| ReID vs bạn | 0.703 | 0.645 | 0.768 | 0.873 | 0.847 | 0.699 | 0.862 | 91 | 104 | 1 |

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

MOTA (0.864) của tôi thấp hơn IDF1 (0.936). Nếu MOTA cao mà IDF1 thấp thì có nghĩa là phát hiện đúng vật thể (Detector tốt) nhưng tracker bị gán lộn ID (Association kém). MOTA không phạt nặng lỗi ID vì công thức của nó trừ điểm đều cho FP, FN và IDSW, mà trong một video dài, số lượng FN và FP thường áp đảo số lượng IDSW (lỗi nhảy ID chỉ đếm 1 lần khi xảy ra), nên IDSW ít ảnh hưởng đến tổng MOTA.

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để giải thích treatment tốt hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai tracker implementation khác.**

- **IDF1:** Tăng từ 0.875 (ByteTrack) lên 0.900 (ReID).
- **AssA:** Tăng từ 0.776 lên 0.820.
- **IDSW:** Giữ nguyên ở mức 2.
ReID giúp giữ định danh (AssA, IDF1) tốt hơn rõ rệt. Tuy nhiên, lưu ý đây là sự so sánh cấp hệ thống (system-level) nên không thể quy toàn bộ chênh lệch này là do ReID, vì implementation của ByteTrack và BoT-SORT là khác nhau. (Sequence ví dụ: Track 4 và 7 bị tách làm 2 mảnh ở ByteTrack nhưng BoT-SORT nối liền được tốt hơn nhờ đặc trưng hình ảnh).

**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

- **DetA:** Tăng mạnh từ 0.649 lên 0.711.
- **FN:** Giảm mạnh từ 54 xuống 26.
- **FP:** Tăng nhẹ từ 88 lên 91.
Lỗi còn lại chủ yếu là do Detector (đặc biệt là FP - bắt nhầm vật thể tĩnh hoặc nền). Sự cải thiện mạnh ở FN cho thấy ReID và BoT-SORT vớt lại được nhiều bounding box khó (khi xe bị che) hơn so với ByteTrack.

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

Model bắt dư 43 frame cho ID 7 (từ frame 16-116), đây là đoạn ReID khoanh nhầm một vật thể tĩnh/nền (ảo ảnh/biển báo) thành xe. Trong nhãn của tôi không có vì đó không phải là xe thật.

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

Tại frame 87, ID 5 trong nhãn tay của tôi đã có sự nhầm lẫn với ID 17/18 của hệ thống, cho thấy hai bản đã lệch phân định danh xe tại điểm occlusion (che khuất) này. Tôi cần soi lại cách gán ID lúc hai xe đi lướt qua nhau.

## 6. Nếu phải gán thêm 10 clip nữa

Bạn sẽ sửa gì trong `GUIDELINE_MINI.md`, và đổi gì trong quy trình làm việc của mình?

- Bổ sung quy định rõ ràng về **điểm kết thúc track**: phải bấm nút outside ngay tại frame đầu tiên xe rời hoàn toàn khỏi mép khung hình để tránh tạo ra bbox treo.
- Tăng tần suất kiểm tra giữa hai keyframe, đặc biệt với các xe chuyển hướng (đi cong) để tránh hiện tượng Bbox trôi (IoU rớt xuống dưới 0.6).

## 7. Tệp đã nộp

- [X] `annotations/clip_01/gt.txt`
- [X] `annotations/clip_02/gt.txt`
- [X] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json`
- [X] `GUIDELINE_MINI.md` đã điền
- [X] `outputs/eval_vs_gold.json`
- [X] `outputs/model_bytetrack_clip_01.txt`
- [X] `outputs/model_reid_clip_01.txt`
- [X] `outputs/model_run_config.json`
- [X] `outputs/eval_bytetrack_vs_gold.json`, `outputs/eval_reid_vs_gold.json`, `outputs/eval_reid_vs_me.json`
- [X] `reports/review_partner.md`
- [X] `reports/REPORT.md` (file này)
