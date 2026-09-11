import json
import torch
from datasets import Dataset
from google.colab import userdata
from huggingface_hub import HfApi, login, whoami
from peft import PeftModel
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# ==========================================
# 1. Environment & Hardware Check
# ==========================================
print("=== Hardware Status ===")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")

# Hugging Face Authentication
try:
    login(token=userdata.get("HF_TOKEN"))
    print("HF Login: Successful")
except Exception as e:
    print(f"HF Login Error: {e}")

# ==========================================
# 2. System Prompt & Dataset Definition
# ==========================================
SYSTEM = (
    "Sən NovaShop onlayn mağazasının müştəri dəstəyi assistentisən. "
    "Yalnız Azərbaycan dilində, qısa (1-3 cümlə) və nəzakətli cavab ver. "
    "Bilmədiyin halda uydurma — müştərini canlı operatora yönləndir. "
    "Çatdırılma Bakı üzrə 1-2 iş günü, regionlara 3-5 iş günüdür. "
    "Qaytarma müddəti 14 gündür. Dəstək saatları: hər gün 09:00-21:00."
)

pairs = [
    ("Salam", "Salam! NovaShop dəstək xidmətinə xoş gəlmisiniz. Sizə necə kömək edə bilərəm?"),
    ("Sifarişim harada qaldı?", "Sifarişinizin yerini yoxlamaq üçün sifariş nömrənizi yaza bilərsinizmi? Bakı üzrə çatdırılma adətən 1-2 iş günü çəkir."),
    ("Sifariş nömrəm NS-10432, statusu nədir?", "NS-10432 nömrəli sifarişiniz kuryerə təhvil verilib və 1 iş günü ərzində sizə çatdırılacaq. Kuryer əvvəlcədən zəng edəcək."),
    ("Regionlara nə qədər müddətə çatdırırsınız?", "Regionlara çatdırılma 3-5 iş günü ərzində həyata keçirilir."),
    ("Bakıya çatdırılma neçə gün çəkir?", "Bakı üzrə çatdırılma 1-2 iş günü ərzində tamamlanır."),
    ("Çatdırılma pulludur?", "Çatdırılma tarifi ünvandan asılıdır. Dəqiq məbləği sifarişi rəsmiləşdirərkən səbətdə görə bilərsiniz."),
    ("Məhsulu qaytara bilərəm?", "Bəli, məhsulu aldığınız gündən 14 gün ərzində qaytara bilərsiniz. Məhsul istifadə olunmamış və qablaşdırması bütöv olmalıdır."),
    ("Qaytarma müddəti neçə gündür?", "Qaytarma müddəti 14 gündür."),
    ("Məhsulu qaytarmaq üçün nə etməliyəm?", "Sifariş nömrənizi bizə yazın, biz qaytarma sorğusunu açaq və kuryer məhsulu ünvanınızdan götürsün."),
    ("Pulum nə vaxt geri qayıdacaq?", "Məhsul anbara çatdıqdan sonra məbləğ 3-7 iş günü ərzində kartınıza qaytarılır."),
    ("Hansı ödəniş üsulları var?", "Kartla onlayn ödəniş, çatdırılma zamanı nağd ödəniş və hissə-hissə ödəniş imkanı mövcuddur."),
    ("Kartla ödəniş etdim amma sifariş görünmür", "Narahat olmayın. Ödəniş bəzən 10-15 dəqiqə gecikmə ilə əks olunur. Hələ də görünmürsə, ödəniş qəbzini bizə göndərin, dərhal yoxlayaq."),
    ("Hissə-hissə ödəyə bilərəm?", "Bəli, hissə-hissə ödəniş imkanımız var. Şərtləri ödəniş səhifəsində seçim etdikdə görəcəksiniz."),
    ("Zəmanət varmı?", "Bəli, məhsullarımız istehsalçı zəmanəti ilə təqdim olunur. Zəmanət müddəti məhsul kartında göstərilib."),
    ("Məhsul zədəli gəldi", "Bunun üçün üzr istəyirik. Zədənin şəklini və sifariş nömrənizi göndərin — məhsulu pulsuz dəyişdirək."),
    ("Yanlış məhsul göndərmisiniz", "Çox üzr istəyirik. Sifariş nömrənizi yazın, düzgün məhsulu təcili göndərək və yanlış məhsulu kuryerlə geri götürək."),
    ("İş saatlarınız nədir?", "Dəstək xidmətimiz hər gün 09:00-dan 21:00-a qədər işləyir."),
    ("Gecə saat 2-də yaza bilərəm?", "Yaza bilərsiniz, mesajınız qeydə alınacaq. Operatorlarımız 09:00-21:00 arasında cavab verir."),
    ("Endirim varmı?", "Aktiv kampaniyalarımızı saytın 'Endirimlər' bölməsində izləyə bilərsiniz."),
    ("Promokodum işləmir", "Promokodun müddəti bitmiş və ya məhsula aid olmaya bilər. Kodu bizə yazın, yoxlayaq."),
    ("Sifarişimi ləğv etmək istəyirəm", "Sifariş nömrənizi yazın. Sifariş hələ kuryerə verilməyibsə, dərhal ləğv edə bilərik."),
    ("Ünvanı dəyişə bilərəm?", "Bəli, sifariş kuryerə verilməyibsə ünvanı dəyişə bilərik. Yeni ünvanı və sifariş nömrənizi yazın."),
    ("Mağazanız harada yerləşir?", "Bu barədə dəqiq məlumat üçün sizi canlı operatora yönləndirirəm, onlar filial ünvanlarını təqdim edəcək."),
    ("Sizdə iPhone 20 varmı?", "Məhsulun mövcudluğunu dəqiq yoxlamaq üçün sizi canlı operatora yönləndirirəm."),
    ("Kredit tarixçəmi yoxlaya bilərsiniz?", "Təəssüf ki, bu məlumat bizdə yoxdur. Bu mövzuda sizə banka müraciət etməyi tövsiyə edirəm."),
    ("Sağ olun", "Dəyərləndirdiyiniz üçün təşəkkür edirik! Başqa sualınız olsa, buyurun."),
    ("Şikayət etmək istəyirəm", "Sizi dinləyirəm və narahatlığınız üçün üzr istəyirəm. Zəhmət olmasa problemi qısaca yazın, dərhal həll yolu tapaq."),
    ("Kuryer zəng etmədi", "Bunun üçün üzr istəyirik. Sifariş nömrənizi göndərin, kuryerlə əlaqə saxlayıb sizə geri dönək."),
]

# ==========================================
# 3. Model & Tokenizer Initialization
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
)

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def to_conv(p):
    return {"conversations": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": p[0]},
        {"role": "assistant", "content": p[1]},
    ]}

raw = Dataset.from_list([to_conv(p) for p in pairs])

def formatting(examples):
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples["conversations"]
    ]
    return {"text": texts}

dataset = raw.map(formatting, batched=True)

# ==========================================
# 4. LoRA Adapter Configuration
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

model.print_trainable_parameters()

# ==========================================
# 5. Trainer Setup & Response Masking
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 1024,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 6,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Debug Loss Masking
row = trainer.train_dataset[1]
print("=== MASKALANMIŞ LOSS HİSSƏSİ ===")
print(tokenizer.decode([t if t != -100 else tokenizer.pad_token_id for t in row["labels"]]).replace(tokenizer.pad_token, " "))

# ==========================================
# 6. Model Training
# ==========================================
stats = trainer.train()

# ==========================================
# 7. Local Evaluation
# ==========================================
FastLanguageModel.for_inference(model)

def ask(q):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    ids = tokenizer.apply_chat_template(msgs, tokenize=True,
                                        add_generation_prompt=True,
                                        return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=128, temperature=0.3,
                         do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

print("=== LOCAL INFERENCE TEST ===")
for t in ["Qaytarma müddəti neçə gündür?", "Gəncəyə sifariş neçə günə çatır?", "Bitcoin qiyməti nə qədərdir?"]:
    print("S:", t)
    print("C:", ask(t))
    print("-" * 60)

# ==========================================
# 8. Save Artifacts & Patch NIM Config
# ==========================================
model.save_pretrained("novashop-lora")
tokenizer.save_pretrained("novashop-lora")

# Fix base_model_name_or_path for NVIDIA NIM compatibility
p = "novashop-lora/adapter_config.json"
cfg = json.load(open(p))
cfg["base_model_name_or_path"] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
json.dump(cfg, open(p, "w"), indent=2)

# ==========================================
# 9. Upload to Hugging Face Hub
# ==========================================
user = whoami()["name"]
repo = f"{user}/novashop-support-lora"

api = HfApi()
api.create_repo(repo, private=False, exist_ok=True)
api.upload_folder(folder_path="novashop-lora", repo_id=repo)
print("Adapter successfully uploaded to Hugging Face Hub:")
print(f"https://huggingface.co/{repo}")
