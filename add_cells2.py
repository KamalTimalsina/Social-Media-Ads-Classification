import json

with open('data.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# More cells for log transformation, scaling, and model training
new_cells = [
    {
        "cell_type": "code",
        "id": "log_transformation",
        "metadata": {},
        "source": [
            "# Log Transformation for highly skewed features\n",
            "print('=== Applying Log Transformation to Highly Skewed Features ===')\n",
            "\n",
            "# Create a copy for transformation\n",
            "df_transformed = df_reduced.copy()\n",
            "\n",
            "# Apply log1p transformation to highly skewed features\n",
            "# log1p = log(1 + x) which handles 0 and negative values\n",
            "for col in highly_skewed:\n",
            "    if (df_transformed[col] >= 0).all():\n",
            "        df_transformed[col] = np.log1p(df_transformed[col])\n",
            "        print(f'Applied log1p transformation to {col}')\n",
            "        print(f'  Skewness before: {skewness_df.loc[col, \"Skewness\"]:.4f}')\n",
            "        print(f'  Skewness after: {df_transformed[col].skew():.4f}')\n",
            "\n",
            "print('\\nTransformation complete!')\n",
            "\n",
            "# Plot transformed distributions\n",
            "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
            "axes = axes.flatten()\n",
            "\n",
            "feature_cols = [col for col in df_transformed.columns if col != 'Purchased']\n",
            "for idx, col in enumerate(feature_cols):\n",
            "    if idx < len(axes):\n",
            "        axes[idx].hist(df_transformed[col], bins=30, edgecolor='black', alpha=0.7, color='green')\n",
            "        new_skew = df_transformed[col].skew()\n",
            "        axes[idx].set_title(f'{col} (Skew: {new_skew:.3f})')\n",
            "        axes[idx].set_xlabel('Value')\n",
            "        axes[idx].set_ylabel('Frequency')\n",
            "        axes[idx].grid(alpha=0.3)\n",
            "\n",
            "for idx in range(len(feature_cols), len(axes)):\n",
            "    axes[idx].set_visible(False)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.suptitle('Feature Distributions AFTER Log Transformation', fontsize=14, fontweight='bold', y=1.00)\n",
            "plt.show()"]
    },
    {
        "cell_type": "markdown",
        "id": "scaling_section",
        "metadata": {},
        "source": ["## J. Scaling and Normalization\n",
                   "\n",
                   "Apply MinMaxScaler to normalize numeric features to [0, 1] range and save the scaler weights."]
    },
    {
        "cell_type": "code",
        "id": "scaling_normalization",
        "metadata": {},
        "source": [
            "import pickle\n",
            "from sklearn.preprocessing import MinMaxScaler\n",
            "\n",
            "print('=== Scaling & Normalization with Weight Saving ===')\n",
            "\n",
            "# Separate features and target\n",
            "feature_cols = [col for col in df_transformed.columns if col != 'Purchased']\n",
            "X = df_transformed[feature_cols].copy()\n",
            "y = df_transformed['Purchased'].copy()\n",
            "\n",
            "print(f'Features to scale: {feature_cols}')\n",
            "print(f'Feature statistics before scaling:')\n",
            "print(X.describe().round(4))\n",
            "\n",
            "# Initialize and fit MinMaxScaler\n",
            "scaler = MinMaxScaler(feature_range=(0, 1))\n",
            "X_scaled = scaler.fit_transform(X)\n",
            "\n",
            "# Convert to DataFrame for easier handling\n",
            "X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)\n",
            "\n",
            "print(f'\\nFeature statistics after MinMaxScaler (0-1 range):')\n",
            "print(X_scaled_df.describe().round(4))\n",
            "\n",
            "# Save scaler weights for demo purposes\n",
            "scaler_path = 'scaler_weights.pkl'\n",
            "with open(scaler_path, 'wb') as f:\n",
            "    pickle.dump(scaler, f)\n",
            "print(f'\\nScaler weights saved to: {scaler_path}')\n",
            "print(f'Scaler components saved:')\n",
            "print(f'  - Feature min values: {scaler.data_min_}')\n",
            "print(f'  - Feature max values: {scaler.data_max_}')\n",
            "print(f'  - Scale (max - min): {scaler.data_max_ - scaler.data_min_}')\n",
            "\n",
            "# Verify scaling\n",
            "print(f'\\nVerifying all values are in [0, 1] range:')\n",
            "print(f'  - Min value in scaled features: {X_scaled.min():.6f}')\n",
            "print(f'  - Max value in scaled features: {X_scaled.max():.6f}')\n",
            "print(f'  - All values in valid range: {(X_scaled.min() >= 0) and (X_scaled.max() <= 1)}')"]
    },
    {
        "cell_type": "code",
        "id": "post_scaling_normality",
        "metadata": {},
        "source": [
            "print('=== Normality Test: Skewness Analysis (AFTER Scaling) ===')\n",
            "\n",
            "# Calculate skewness for scaled features\n",
            "post_scale_skewness = {}\n",
            "for col in feature_cols:\n",
            "    skew = X_scaled_df[col].skew()\n",
            "    kurtosis = X_scaled_df[col].kurtosis()\n",
            "    post_scale_skewness[col] = {'Skewness': skew, 'Kurtosis': kurtosis}\n",
            "\n",
            "post_scale_df = pd.DataFrame(post_scale_skewness).T.round(4)\n",
            "print('Skewness and Kurtosis after scaling:')\n",
            "print(post_scale_df)\n",
            "\n",
            "# Compare before and after\n",
            "print('\\nComparison - Skewness Change:')\n",
            "print('Feature                 Before Scaling      After Scaling       Change')\n",
            "print('-' * 75)\n",
            "for col in feature_cols:\n",
            "    before = skewness_df.loc[col, 'Skewness'] if col in skewness_df.index else df_reduced[col].skew()\n",
            "    after = post_scale_df.loc[col, 'Skewness']\n",
            "    change = after - before\n",
            "    print(f'{col:20} {before:8.4f}           {after:8.4f}           {change:+.4f}')\n",
            "\n",
            "print('\\nNote: MinMaxScaler preserves skewness shape but rescales the range.')\n",
            "print('Log transformation was applied before scaling to normalize highly skewed features.')"]
    },
    {
        "cell_type": "markdown",
        "id": "model_training_section",
        "metadata": {},
        "source": ["## L. Model Training\n",
                   "\n",
                   "Train multiple classification models and evaluate performance."]
    },
    {
        "cell_type": "code",
        "id": "train_test_split",
        "metadata": {},
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "print('=== Train-Test Split Setup ===')\n",
            "\n",
            "# Split data into training and testing sets (75-25 split)\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X_scaled_df, y, test_size=0.25, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f'Total samples: {len(X_scaled_df)}')\n",
            "print(f'Training set: {len(X_train)} samples ({len(X_train)/len(X_scaled_df)*100:.1f}%)')\n",
            "print(f'Testing set: {len(X_test)} samples ({len(X_test)/len(X_scaled_df)*100:.1f}%)')\n",
            "\n",
            "print(f'\\nTarget class distribution in training set:')\n",
            "print(y_train.value_counts())\n",
            "print(f'\\nTarget class distribution in testing set:')\n",
            "print(y_test.value_counts())\n",
            "\n",
            "print(f'\\nClass balance maintained (stratified split confirmed)')"]
    }
]

# Add new cells to notebook
for cell in new_cells:
    nb['cells'].append(cell)

# Write updated notebook
with open('data.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(new_cells)} cells for transformations, scaling, and model setup")
print(f"Total cells now: {len(nb['cells'])}")
