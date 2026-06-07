# 文字資料清洗 + 重切 稽核報告

- 來源：text_train.jsonl + text_valid.jsonl（共 197 筆，去除重複 7 筆）
- 重切後 train：148 筆（釣魚 32 / 正常 116）
- 重切後 valid：42 筆（釣魚 18 / 正常 24）
- 跨 split 網域洩漏：0（目標 0）

## 自動修正的錯標（已套用）
- `https://barclays-banking.net/landing/form/81ff82f0-3bb1-4f13-9ce2-8447f3646657`：risk 0.58 → 0.88　Brand-domain mismatch: impersonates Barclays bank on a non-official domain.

## 灰區待人工確認（0.4 ≤ risk < 0.7，未改動）
- risk=0.67 `http://www.hayatseninleguzel.store/auth`　['Uses a low-reputation or commonly abused TLD (.store).']
- risk=0.67 `https://airbnbclone-jrg8daobo-yusef-mohamed.vercel.app/`　['Hosted on a free or instant deployment platform (vercel.app).']
- risk=0.63 `https://rafizaman577.github.io/Amazon/checkout.html`　['Hosted on a free or instant deployment platform (github.io).']
- risk=0.63 `https://newdapp-swap.netlify.app/`　['Hosted on a free or instant deployment platform (netlify.app).']
- risk=0.59 `http://safari.spahotel.guru/`　['Uses a low-reputation or commonly abused TLD (.guru).']
- risk=0.58 `https://www.roblox.com/Login`　['URL path suggests credential, wallet, payment, or claim flow.']
- risk=0.58 `http://geminihospitality.in/new/wp-content/Post/NV6588123/`　['Visible text contains credential, verification, reward, or urgency phrases.']
- risk=0.54 `https://com--dubai-booking.webflow.io/`　['Host contains many hyphens, a common cybersquatting pattern.']
- risk=0.54 `https://www.365bet881.com/`　['Host contains many digits, consistent with disposable domains.']
- risk=0.50 `https://stop.blogbeastit.info/?app_vl=ZYFwlHFpbWKEmLqxy5qmnnx0YsC2wa-TpaiVYsBxj2phmqOgnLFwrYw&e=user@example.com&sui=%7Bsui%7D&fn=&ln=&p=&z=PL-PLNAMERAW-2705-ind7`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://loss.blogbeastit.info/?app_vl=ZYFwlHFpbWKEmLqxy5qmnnx0YsC2wa-TpaiVYsBxj2phmqOgnLFwrYw&e=solelx@fc6ed5fae4a56ff9909b3bbb3f96e465a2a5.com&sui=%7Bsui%7D&fn=Kateryna&ln=Morawik&p=&z=PLNAMERAW-2905-digi9`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://www.open-bet365.com.cn/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://dsignpost.com/infospage.php`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://life-bet365.com.cn/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://we-only.statichost.page/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `http://aktifkndannaxpaylter.tubersis.biz.id/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://tohuusin.info/?app_vl=Zn9wlHFpbWKEmLqxy5qmnnx0YsC2wa-TpaiVYsBxj2phmqOgnLFwrYw&e=norynlab@89c7ee1d02f23d413d4b6b0dd215632b69a0.org&sui=%7Bsui%7D&fn=&ln=&p=&z=chnamecl280526con1`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://www.roblox.com/games/92416421522960/Slime-RNG`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://seal-app-gvgyd.ondigitalocean.app/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://www.pekinshuho.com/imwy/501.html`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `https://jellyfish-app-jsmsz.ondigitalocean.app/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
- risk=0.50 `http://www.yh9512.com/`　['No strong phishing phrase was visible; URL and text evidence is limited.']
