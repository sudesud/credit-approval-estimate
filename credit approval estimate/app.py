import streamlit as st
import numpy as np
import joblib

# Model ve scaler'ı yükle
saved_data = joblib.load('eniyi.joblib')
model = saved_data['model']
scaler = saved_data['scaler']

# Eğitim sırasında kullanılan etiket kodlaması
education_mapping = {"Lise": 0, "Üniversite": 1, "Yüksek Lisans": 2, "Doktora": 3}

st.title("Kredi Onaylama Tahmini")
st.write("Lütfen aşağıdaki bilgileri doldurun:")

# Kullanıcı girdileri
gelir = st.number_input("Gelir", min_value=0.0, step=1000.0)
yaş = st.number_input("Yaş", min_value=18, step=1)
kredi_skoru = st.number_input("Kredi Skoru", min_value=0.0, step=1.0)
borç_oranı = st.number_input("Borç Oranı", min_value=0.0, max_value=1.0, step=0.01)
çalışma_süresi = st.number_input("Çalışma Süresi", min_value=0, step=1)
eğitim_seviyesi = st.selectbox("Eğitim Seviyesi", ["Lise", "Üniversite", "Yüksek Lisans", "Doktora"])

# Tahmin butonu
if st.button("Tahmin Et"):
    # Eğitim seviyesi sayısal değere dönüştürülüyor
    eğitim_kodu = education_mapping[eğitim_seviyesi]

    # Girdi verisi oluşturuluyor
    input_data = np.array([[gelir, yaş, kredi_skoru, borç_oranı, çalışma_süresi, eğitim_kodu]])
    
    # Veriyi ölçeklendirme
    input_scaled = scaler.transform(input_data)
    
    # Model ile tahmin
    prediction = model.predict(input_scaled)
    
    # Tahmin sonucunu göster
    if prediction[0] == 1:
        st.success("Kredi başvurunuz onaylanabilir.")
    else:
        st.error("Kredi başvurunuz reddedilebilir.")
