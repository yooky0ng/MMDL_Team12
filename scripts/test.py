from datasets import load_dataset

ds = load_dataset(
    "MMMU/MMMU",
    "Accounting",
    split="validation",
    revision="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
)

print(ds)
print(ds[0])