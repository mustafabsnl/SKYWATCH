SKYWATCH — Proje Durumu
🏗️ Altyapı ve Sistem Mimarisi — Tamamlandı
Projenin temel iskelet yapısı eksiksiz kurulmuştur. Konfigürasyon sistemi (config.yaml), olay loglama sistemi (EventLogger), GPU kurulum modülü ve modüler dizin yapısı çalışır haldedir. Sistem, threaded (çok iş parçacıklı) mimari üzerine inşa edilmiş olup görüntü okuma, işleme ve ekranda gösterme adımları birbirinden bağımsız çalışmaktadır. Bu sayede herhangi bir modül yavaşladığında görüntü donmamaktadır.

📹 Kamera Yönetimi — Tamamlandı
CameraManager modülü hem webcam hem de video dosyası kaynaklarını desteklemektedir. Video kaynağının native FPS değeri otomatik okunarak görüntü doğru hızda oynatılmaktadır. Bağlantı kesilmelerinde otomatik yeniden bağlanma mekanizması mevcuttur.

🤖 Yüz Algılama — Geliştirme Aşamasında
InsightFace buffalo_l modeli entegre edilmiş ve NVIDIA GPU (CUDA 12.6) üzerinde çalışır hale getirilmiştir. Frame başına işlem süresi CPU'daki 1500ms seviyesinden GPU ile 50ms seviyesine indirilmiştir. Ancak yüz tanıma doğruluğu ve farklı açı/ışık koşullarındaki performans henüz istenen düzeyde değildir. Eşik değerleri, model parametreleri ve ön işleme adımları üzerinde çalışmalar devam etmektedir.

👤 Takip Sistemi (Tracking) — Geliştirme Aşamasında
DeepSORT algoritması temel düzeyde entegre edilmiş olup kişilere kararlı kimlik numarası (Track ID) atanabilmektedir. Ancak kimlik kararlılığı, takip kaybı durumunda yeniden eşleştirme ve kameralar arası geçiş takibi gibi ileri özellikler henüz geliştirilmektedir. Hareket analiz modülü (hız, yön, bekleme süresi) temel seviyede çalışmakla birlikte davranış tespiti algoritmaları optimize edilmeye devam etmektedir.

⚖️ Karar Motoru — Geliştirme Aşamasında
Sistem kişileri CLEAN, SUSPICIOUS ve WANTED olarak etiketleyebilmekte ve renk kodlu kutuyla göstermektedir. Ancak karar mekanizması henüz basit eşik değerlerine dayanmaktadır. Çoklu faktör ağırlıklandırması, şüpheli davranış senaryoları ve yanlış pozitif oranını düşürecek gelişmiş karar kriterleri üzerinde çalışmalar sürmektedir.

🗄️ Veritabanı — Geliştirme Aşamasında
SQLite tabanlı veritabanı şeması oluşturulmuş; sabıkalı kişi kayıtları, yüz embedding vektörleri ve tespit logları için tablolar hazırlanmıştır. Temel CRUD işlemleri çalışmaktadır. Ancak embedding karşılaştırma performansı, çoklu embedding desteği (aynı kişinin farklı açıdan fotoğrafları) ve tespit geçmişi raporlama özellikleri geliştirilme aşamasındadır.

🖥️ Arayüz (GUI) — Henüz Başlanmadı
Kullanıcı arayüzü (PyQt5) planlanmış ancak henüz geliştirilmemiştir. Bu aşamada sistem terminal üzerinden başlatılmakta, görüntü OpenCV penceresiyle izlenmektedir.

🔍 Aktif Kişi Arama — Henüz Başlanmadı
Fotoğraf yükleyerek canlı kameralarda belirli bir kişiyi arama özelliği altyapıda tanımlanmış, veritabanı şeması hazırlanmış ancak henüz aktif hale getirilmemiştir.