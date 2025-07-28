#!/usr/bin/env python3
# train_liftoff_classifier.py

import os, cv2, random, argparse, pandas as pd, tensorflow as tf
from sklearn.metrics import precision_recall_curve
import numpy as np, os, tensorflow as tf

# ── 1) Data prep: sample 5× more negatives ───────────────────────────────────
def prepare_datasets(video_dir, csv_path, output_dir,
                     img_size=(224,224), val_split=0.2, seed=42):
    rnd = random.Random(seed)
    df = pd.read_csv(csv_path)
    df['Video_Name'] = df['Video_Name'].apply(
        lambda v: v if v.lower().endswith('.mp4') else v+'.mp4'
    )

    pos_samples, neg_samples = [], []
    for video_name, grp in df.groupby("Video_Name"):
        path = os.path.join(video_dir, video_name)
        if not os.path.isfile(path):
            print(f"⚠️ Missing {path}")
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        positives = sorted(grp["LiftOff_Frame"].astype(int))
        # 5× negatives per positive
        all_neg = list(set(range(total)) - set(positives))
        negs = rnd.sample(all_neg, k=len(positives)*5)

        pos_samples += [(video_name,f) for f in positives]
        neg_samples += [(video_name,f) for f in negs]

    # global split
    rnd.shuffle(pos_samples); rnd.shuffle(neg_samples)
    split_p = int(len(pos_samples)*(1-val_split))
    split_n = int(len(neg_samples)*(1-val_split))
    splits = {
        "train": {"liftoff": pos_samples[:split_p],
                  "not_liftoff": neg_samples[:split_n]},
        "val":   {"liftoff": pos_samples[split_p:],
                  "not_liftoff": neg_samples[split_n:]}
    }

    # write out frames
    for split, labels in splits.items():
        for label, samples in labels.items():
            out_dir = os.path.join(output_dir, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for video_name, fidx in samples:
                cap = cv2.VideoCapture(os.path.join(video_dir,video_name))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read(); cap.release()
                if not ret: continue
                img = cv2.resize(frame, img_size)
                fn = f"{os.path.splitext(video_name)[0]}_{label}_{fidx}.jpg"
                path = os.path.join(out_dir, fn)
                cv2.imwrite(path, img)

    # ── 2) Build tf.data datasets ──────────────────────────────────────────────
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(output_dir,"train"),
        labels="inferred", label_mode="binary",
        image_size=img_size, batch_size=32, shuffle=True, seed=seed
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(output_dir,"val"),
        labels="inferred", label_mode="binary",
        image_size=img_size, batch_size=32, shuffle=False
    )

    # ── 3) Data augmentation on-the-fly ────────────────────────────────────────
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomRotation(0.1),
      tf.keras.layers.RandomZoom(0.2),
      tf.keras.layers.RandomBrightness(0.2),    # new
      tf.keras.layers.RandomContrast(0.2), 
    ])
    train_ds = train_ds.map(
      lambda x,y: (data_augmentation(x, training=True), y),
      num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_ds, val_ds


# ── 4) Build & compile the base model ────────────────────────────────────────
def build_model(input_shape=(224,224,3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(base.input, out)
    model.compile(
      optimizer="adam",
      loss="binary_crossentropy",
      metrics=["accuracy"]
    )
    return model


# ── 5) Main: train with class weights, then fine-tune ───────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", required=True)
    p.add_argument("--csv",       required=True)
    p.add_argument("--out_dir",   default="data")
    p.add_argument("--model_out", default="liftoff_image_classifier.h5")
    p.add_argument("--img_size",  type=int, nargs=2, default=[224,224])
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--epochs",    type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    train_ds, val_ds = prepare_datasets(
        args.video_dir, args.csv, args.out_dir,
        img_size=tuple(args.img_size),
        val_split=args.val_split,
        seed=args.seed
    )

    print("📊 Datasets ready. Building model…")
    model = build_model(input_shape=tuple(args.img_size)+(3,))

        # compute class weights by counting files on disk
    import glob
    n_pos = len(glob.glob(os.path.join(args.out_dir, "train", "liftoff", "*.jpg")))
    n_neg = len(glob.glob(os.path.join(args.out_dir, "train", "not_liftoff", "*.jpg")))
    total = n_pos + n_neg
    class_weight = {
        0: total / (2 * n_neg),  # weight for “not_liftoff”
        1: total / (2 * n_pos)   # weight for “liftoff”
    }


    print("🏋️‍♀️ Training head (with class weights)…")
    model.fit(
      train_ds, validation_data=val_ds,
      epochs=args.epochs,
      class_weight=class_weight
    )

    # ── 6) Fine‐tune top MobileNet block ──────────────────────────────────────
    print("🔓 Unfreezing last 30 layers for fine‐tuning…")
    model.trainable = True

    for layer in model.layers[:-30]:
       layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
      loss="binary_crossentropy",
      metrics=["accuracy"]
    )
    
    model.fit(
      train_ds, validation_data=val_ds,
      epochs=3
    )
    
    print(f"💾 Saving model to {args.model_out}")
    model.save(args.model_out)
    print("✅ Done.")

if __name__=="__main__":
    main()
