# Proje Kod Standartları ve Mimari Kurallar

1. State Management:
   - Projede durum yönetimi için kesinlikle BLoC veya Cubit kullanılmalıdır. 
   - UI widget'ları içerisinde direkt olarak durum tutulmamalı (setState kaçınılmalı) ve iş mantığı yazılmamalıdır.

2. Network ve API Requests:
   - Tüm HTTP istekleri Dio kütüphanesi üzerinden yapılmalıdır.
   - API çağrıları doğrudan UI'dan değil, ilgili Repository sınıfı üzerinden yürütülmelidir.
   - Hata yönetiminde `print()` kullanımı kesinlikle yasaktır, loglama için `developer.log()` kullanılmalıdır.

3. Isimlendirme ve Klasör Yapısı:
   - Sınıf isimleri PascalCase, dosya isimleri snake_case olmalıdır.
   - Feature-first klasör yapısı uygulanmalıdır (örn: `lib/features/auth/presentation/`).