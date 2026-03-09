import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def process_svi_ema_workflow(input_file, output_file):
    print(f"🚀 Start processing: {input_file}")
    df = pd.read_csv(input_file)

    # ---------------------------------------------------------
    # 1. Configure fields
    # ---------------------------------------------------------
    # Sensor and SVI core fields
    sensor_cols = ['PM2_5_calibrated_mean', 'Temperature_mean', 'Humidity_mean']
    svi_cols = [
        'road_percent', 'sidewalk_percent', 'building_percent', 'wall_percent',
        'fence_percent', 'pole_percent', 'traffic light_percent', 'traffic sign_percent',
        'vegetation_percent', 'terrain_percent', 'sky_percent', 'person_percent',
        'rider_percent', 'car_percent', 'truck_percent', 'bus_percent',
        'train_percent', 'motorcycle_percent', 'bicycle_percent'
    ]
    all_target_cols = sensor_cols + svi_cols

    # Spatiotemporal feature fields (referencing your provided code names)
    geo_cols = ['lon_gtt', 'lat_gtt']

    # ---------------------------------------------------------
    # 2. Outlier flagging phase (key: first invalidate, then reconstruct)
    # ---------------------------------------------------------
    print("🧹 Identifying extreme outliers and setting them to NaN...")

    # A. Sensor 3.0 IQR (two-sided cleaning)
    for col in sensor_cols:
        Q1, Q3 = df[col].quantile(0.15), df[col].quantile(0.85)
        IQR = Q3 - Q1
        df.loc[(df[col] < (Q1 - 3.0 * IQR)) | (df[col] > (Q3 + 3.0 * IQR)), col] = np.nan

    # B. Sky physical threshold anomaly (>40% treated as bad sample, entire row of SVI invalidated)
    df.loc[df['sky_percent'] > 40, svi_cols] = np.nan
    #df.loc[df['building_percent'] > 80, svi_cols] = np.nan
    df.loc[(df['building_percent'] > 40) & (df['sky_percent'] < 10), svi_cols] = np.nan
    df.loc[(df['sky_percent'] > 30) & (df['building_percent'] < 10), svi_cols] = np.nan
    df.loc[(df['vegetation_percent'] > 40) & (df['road_percent'] < 10), svi_cols] = np.nan
    # C. Other SVI fields 3.0 IQR (upper bound only, preserving zero values)
    for col in svi_cols:#for col in [c for c in svi_cols if c not in ('sky_percent')]:#for col in svi_cols:
        Q1, Q3 = df[col].quantile(0.15), df[col].quantile(0.85)
        IQR = Q3 - Q1
        df.loc[df[col] > (Q3 + 3.0 * IQR), col] = np.nan

    # ---------------------------------------------------------
    # 3. Preprocessing (time conversion and sorting)
    # ---------------------------------------------------------
    df['_calc_ts'] = pd.to_datetime(df['timestamp'])
    df['_calc_ts_numeric'] = df['_calc_ts'].astype(np.int64) // 10 ** 9
    # Strictly sort by PID -> emaid -> time to ensure correct interpolation direction
    df = df.sort_values(by=['PID', 'emaid', '_calc_ts'])

    # ---------------------------------------------------------
    # 4. Phase 1: Intra-group self-recovery (hierarchical linear interpolation)
    # ---------------------------------------------------------
    print("📈 Performing intra-group (PID + emaid) linear interpolation...")
    for col in all_target_cols:
        # Linear imputation within the same emaid session; limit_direction='both' handles head/tail
        df[col] = df.groupby(['PID', 'emaid'])[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

    # ---------------------------------------------------------
    # 5. Phase 2: Spatiotemporal KNN recovery (global imputation)
    # ---------------------------------------------------------
    if df[all_target_cols].isnull().any().any():
        print("🤖 Running KNN imputation for gaps not covered by interpolation...")

        # Prepare KNN features (Time + GPS)
        feature_cols = ['_calc_ts_numeric'] + geo_cols
        # Fill GPS NaN values with median to prevent KNN errors
        df[geo_cols] = df[geo_cols].fillna(df[geo_cols].median())

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])

        # Prepare and normalize targets
        targets = df[all_target_cols].values
        t_min, t_max = np.nanmin(targets, axis=0), np.nanmax(targets, axis=0)
        t_range = np.where((t_max - t_min) == 0, 1, t_max - t_min)
        targets_scaled = (targets - t_min) / t_range

        # KNN execution (n_neighbors=5, distance-weighted)
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        combined_data = np.hstack([X_scaled, targets_scaled])
        imputed_data = imputer.fit_transform(combined_data)

        # Inverse transform to restore original scale
        res_targets = imputed_data[:, len(feature_cols):] * t_range + t_min
        df_knn = pd.DataFrame(res_targets, columns=all_target_cols, index=df.index)

        # Only fill values that are still missing
        df[all_target_cols] = df[all_target_cols].fillna(df_knn)

    # ---------------------------------------------------------
    # 6. Finalize and save
    # ---------------------------------------------------------
    # Remove helper columns and restore original sort order (optional)
    df.drop(columns=['_calc_ts', '_calc_ts_numeric'], inplace=True)

    df.to_csv(output_file, index=False)
    print(f"✅ Processing complete! High-quality data saved to: {output_file}")


if __name__ == "__main__":
    input_p = r"C:\Files\GLAN_DATA\first_wave\EMA30min\ema_30gps_svisegm.csv"
    output_p = r"C:\Files\GLAN_DATA\first_wave\EMA30min\ema_30gps_svisegm_new.csv"
    process_svi_ema_workflow(input_p, output_p)