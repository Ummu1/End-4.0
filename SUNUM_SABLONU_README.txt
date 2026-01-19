========================================================================
PROJE SUNUM VE TEKNİK RAPOR DOSYASI
(Bu dosyayı doldurup proje klasörünüzün içine ekleyiniz.)
========================================================================

------------------------------------------------------------------------
1. PROJE ÖZETİ
------------------------------------------------------------------------
Ev ve daire fiyatlarını tahmin eden bir yapay zeka modeli geliştirdik. Kullanıcılar, istedikleri ilçe ve mahalleyi seçerek ilanların uygun fiyatlı mı yoksa pahalı mı olduğunu hızlıca görebilecekler. Böylece emlak yatırım kararlarını daha bilinçli verebilecekler.


------------------------------------------------------------------------
2. KURULUM VE ÇALIŞTIRMA (How to Run)
------------------------------------------------------------------------
(Jüri üyeleri projenizi kendi bilgisayarlarında nasıl çalıştıracak? Adım adım yazın.)
Örnek:
1. pip install -r requirements.txt
2. python app.py
3. http://localhost:5000/  bu siteye giriniz.
4. İlçe ve mahalle seçimi yaptıktan sonra “Tahmin Et” butonuna tıklayın.


------------------------------------------------------------------------
3. VERİ ÖN İŞLEME YAKLAŞIMI (Data Engineering)
------------------------------------------------------------------------
(Kirli veriyi temizlemek için neler yaptınız? Hangi sütunları elediniz, hangilerini yeni ürettiniz?)
- Temizlenen Alanlar:
Price sütunu TL ve binlik ayırıcıları temizlendi, string → float dönüştürüldü.
Eksik fiyat değerleri kaldırıldı.
Kategorik sütunlardaki eksik değerler Unknown ile dolduruldu.
- Eklenen Yeni Özellikler (Feature Engineering):
Yok (Mevcut verilerden ek özellik oluşturulmadı).
- Seçilen Kritik Özellikler (Feature Selection):
District, Neighborhood, m² (Gross), m² (Net), Number of rooms, Building Age, Floor location, Number of floors, Heating, Number of bathrooms, Balcony, Furnished, Using status, Available for Loan, Front South, Elevator, Parking Lot, Air conditioning, Furniture, Swimming Pool (Open), Sea, Nature, Metro, Bus stop, Hospital, Market, Gym

------------------------------------------------------------------------
4. MODEL MİMARİSİ
------------------------------------------------------------------------
(Hangi algoritmayı kullandınız ve neden?)
- Kullanılan Algoritma: Random Forest Regressor
- Neden bu algoritma?: Özellikler arası karmaşık ilişkileri yakalayabiliyor.

Eksik veri ve kategorik değişkenler ile iyi çalışıyor.

Overfitting’i azaltmak için max_depth ve min_samples parametreleri ayarlandı.

- Elde Edilen Başarı Skoru (RMSE / R-Square):
Train R² Score: 0.92

Test R² Score: 0.87

Train MAE: ~120,000 TL

Test MAE: ~150,000 TL


------------------------------------------------------------------------
5. YATIRIM KARAR MANTIĞI (Business Logic)
------------------------------------------------------------------------
Kullanıcıya "Fırsat" veya "Pahalı" bilgisi, modelin tahmini fiyat ile ilan fiyatı karşılaştırılarak verilir.

- FIRSAT Eşiği:İlan fiyatı < Model Tahmini × 0.90
- PAHALI Eşiği:İlan fiyatı > Model Tahmini × 1.10


------------------------------------------------------------------------
6. SİZİ DİĞERLERİNDEN AYIRAN ÖZELLİK (Unique Value)
------------------------------------------------------------------------
(Projenizde ekstra ne var? Arayüz tasarımı mı, enflasyon hesaplaması mı, farklı bir bakış açısı mı?)
Kullanıcı dostu ve modern HTML arayüz tasarımı.

İlçe → mahalle → ilan fiyat tahmini adımlı seçim sistemi.

Gradient ile görsel olarak öne çıkan tahmin değeri göstergesi.


Fırsat/pahalı uyarısı sayesinde kullanıcılar hızlı yatırım kararı verebiliyor.


