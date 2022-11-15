import timm

model = timm.create_model("resnest14d", pretrained=False, num_classes=2, drop_rate=0.5)

