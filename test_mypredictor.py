from MyPredictor import MyPredictor

p = MyPredictor.from_path(".")
print(p.predict(
    [
        "speero-aiplatform//images/audio_one/M_0030_17y9m_1_32000_2of7_stutter.png",
        "speero-aiplatform//images/audio_one/M_0030_17y9m_1_4000_4of10_stutter.png",
        "speero-aiplatform//images/audio_one/M_0030_17y9m_1_28000_no_stutter.png",
        "speero-aiplatform//images/audio_one/M_0030_17y9m_1_52000_no_stutter.png"
    ]
))
