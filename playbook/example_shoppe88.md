# 釣魚事件處置 Playbook

- 目標網址：`https://shoppe88.com/login`
- 註冊網域：`shoppe88.com`
- 仿冒品牌：Shopee
- 風險分數：0.92

> 自動產生之初步處置建議，請網管/資安人員審閱後執行。

## 1. 網域調查指令 (Whois / DNS)

```bash
whois shoppe88.com                      # 註冊人、註冊商、建立日期（新註冊網域風險高）
dig shoppe88.com +short                 # 解析 IP
dig shoppe88.com NS +short              # 名稱伺服器
nslookup shoppe88.com
curl -sI "https://shoppe88.com/login"                    # 觀察 HTTP header / 重導
```

威脅情報查詢：
- VirusTotal: https://www.virustotal.com/gui/domain/shoppe88.com
- urlscan.io: https://urlscan.io/search/#domain%3Ashoppe88.com
- abuse 聯絡：以 whois 結果中的 registrar abuse email 進行下架檢舉

## 2. 165 檢舉信草稿

主旨：檢舉仿冒 Shopee 釣魚網站 shoppe88.com

內容：
1. 檢舉網址：https://shoppe88.com/login
2. 仿冒對象：Shopee
3. 詐騙手法：仿冒官方頁面，誘導民眾輸入帳號密碼或付款資訊。
4. 請求：懇請協助通報網域下架並列入警示名單，避免民眾受害。

## 3. 使用者防詐宣導訊息

⚠️ 詐騙警示：網站 shoppe88.com 疑似仿冒「Shopee」官方頁面。
請勿在此輸入帳號、密碼或信用卡資訊。Shopee 不會以此類連結要求驗證。
如有疑慮，請撥打 165 反詐騙專線，或直接從官方 App／官網查證。
