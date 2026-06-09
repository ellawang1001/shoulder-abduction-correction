"""
extract_users.py — multi-subject extraction. video-org/user{1..4}/ each holds
ONE round of 30 clips in angle-major order (angle[0,30,60,90,120,150] x
az[0,30,45,60,90]). Each user is calibrated SEPARATELY (different body size /
camera distance), studied side detected per user. Outputs per-user CSVs plus a
combined real_estimates_4users.csv.
"""
import glob, os
import pandas as pd
import pose_extract as PE

CLEAN_ORDER = [(a, z) for a in [0, 30, 60, 90, 120, 150] for z in [0, 30, 45, 60, 90]]
USERS = sorted(os.path.basename(p) for p in glob.glob(os.path.join("video-org", "user*")) if os.path.isdir(p))

dfs = []
for k, u in enumerate(USERS, 1):
    d = os.path.join("video-org", u)
    imgs = sorted(set(glob.glob(os.path.join(d, "*.MOV")) + glob.glob(os.path.join(d, "*.mov"))))
    assert len(imgs) == 30, f"{u}: expected 30, found {len(imgs)}"
    files, info = [], {}
    for i, img in enumerate(imgs):
        ang, az = CLEAN_ORDER[i]
        files.append(img)
        info[img] = {"subject_id": k, "true_angle": ang, "azimuth": az, "rep": 1}
    print(f"\n########## {u} (subject_id={k}) ##########")
    df = PE.run_extract(files, info, f"real_estimates_{u}.csv")
    dfs.append(df)

big = pd.concat(dfs, ignore_index=True)
big.to_csv("real_estimates_4users.csv", index=False)
print(f"\ncombined -> real_estimates_4users.csv: {len(big)} rows, {big.subject_id.nunique()} users")
