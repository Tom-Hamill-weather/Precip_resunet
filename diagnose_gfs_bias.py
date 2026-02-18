"""
diagnose_gfs_bias.py

Quick diagnostic script to test hypotheses about GFS model wet bias.

Usage:
    python diagnose_gfs_bias.py 2025120412 12
"""

import sys
import numpy as np
import _pickle as cPickle
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os

def load_training_data(date, lead):
    """Load training data file."""
    pattern = f'../resnet_data/GRAF_Unet_data_train_{date}_{lead}h.cPick'

    if not os.path.exists(pattern):
        print(f"Training file not found: {pattern}")
        return None

    with open(pattern, 'rb') as f:
        data = {
            'graf': cPickle.load(f),
            'mrms': cPickle.load(f),
            'qual': cPickle.load(f),
            'terdiff_graf': cPickle.load(f),
            'diff': cPickle.load(f),
            'dlon': cPickle.load(f),
            'dlat': cPickle.load(f),
            'init_times': cPickle.load(f),
            'valid_times': cPickle.load(f),
            'gfs_pwat': cPickle.load(f),
            'gfs_r': cPickle.load(f),
            'gfs_cape': cPickle.load(f)
        }

    return data

def analyze_dry_wet_distribution(data):
    """Analyze the distribution of dry vs wet patches."""
    graf = data['graf']
    mrms = data['mrms']
    gfs_pwat = data['gfs_pwat']
    gfs_r = data['gfs_r']
    gfs_cape = data['gfs_cape']

    # Define dry patches
    graf_max = np.max(graf, axis=(1,2))
    mrms_max = np.max(mrms, axis=(1,2))

    dry_graf = graf_max < 0.01
    dry_mrms = mrms_max < 0.01
    light_graf = (graf_max >= 0.01) & (graf_max < 2.5)
    moderate_graf = (graf_max >= 2.5) & (graf_max < 10.0)
    heavy_graf = graf_max >= 10.0

    print("="*70)
    print("TRAINING DATA DISTRIBUTION")
    print("="*70)
    print(f"\nTotal patches: {len(graf)}")
    print(f"\nGRAF-based stratification:")
    print(f"  Dry (< 0.01 mm):        {np.sum(dry_graf):5d} ({100*np.mean(dry_graf):5.1f}%)")
    print(f"  Light (0.01-2.5 mm):    {np.sum(light_graf):5d} ({100*np.mean(light_graf):5.1f}%)")
    print(f"  Moderate (2.5-10 mm):   {np.sum(moderate_graf):5d} ({100*np.mean(moderate_graf):5.1f}%)")
    print(f"  Heavy (>= 10 mm):       {np.sum(heavy_graf):5d} ({100*np.mean(heavy_graf):5.1f}%)")

    print(f"\n  MRMS dry (< 0.01 mm):   {np.sum(dry_mrms):5d} ({100*np.mean(dry_mrms):5.1f}%)")

    # GFS statistics by category
    print("\n" + "="*70)
    print("GFS FEATURE STATISTICS BY PRECIPITATION CATEGORY")
    print("="*70)

    categories = [
        ('Dry', dry_graf),
        ('Light', light_graf),
        ('Moderate', moderate_graf),
        ('Heavy', heavy_graf)
    ]

    for cat_name, mask in categories:
        if np.sum(mask) > 0:
            print(f"\n{cat_name} patches (n={np.sum(mask)}):")
            print(f"  PWAT:  mean={np.mean(gfs_pwat[mask]):6.2f}  median={np.median(gfs_pwat[mask]):6.2f}  "
                  f"std={np.std(gfs_pwat[mask]):6.2f} kg/m²")
            print(f"  RH:    mean={np.mean(gfs_r[mask]):6.2f}  median={np.median(gfs_r[mask]):6.2f}  "
                  f"std={np.std(gfs_r[mask]):6.2f} %")
            print(f"  CAPE:  mean={np.mean(gfs_cape[mask]):6.2f}  median={np.median(gfs_cape[mask]):6.2f}  "
                  f"std={np.std(gfs_cape[mask]):6.2f} J/kg")

    # Check for overlap in GFS feature distributions
    print("\n" + "="*70)
    print("GFS FEATURE OVERLAP ANALYSIS")
    print("="*70)

    if np.sum(dry_graf) > 0 and np.sum(light_graf) > 0:
        dry_pwat = gfs_pwat[dry_graf]
        light_pwat = gfs_pwat[light_graf]

        # What percentile of dry PWAT exceeds median light PWAT?
        overlap_pct = 100 * np.mean(dry_pwat > np.median(light_pwat))
        print(f"\nPWAT overlap:")
        print(f"  {overlap_pct:.1f}% of DRY patches have PWAT > median LIGHT patch")
        print(f"  This suggests: {'HIGH AMBIGUITY' if overlap_pct > 20 else 'Low ambiguity'}")

        # Same for RH
        dry_rh = gfs_r[dry_graf]
        light_rh = gfs_r[light_graf]
        overlap_pct_rh = 100 * np.mean(dry_rh > np.median(light_rh))
        print(f"\nRH overlap:")
        print(f"  {overlap_pct_rh:.1f}% of DRY patches have RH > median LIGHT patch")
        print(f"  This suggests: {'HIGH AMBIGUITY' if overlap_pct_rh > 20 else 'Low ambiguity'}")

def create_gfs_distribution_plot(data, output_file='gfs_distribution_by_precip.png'):
    """Create diagnostic plots showing GFS feature distributions."""
    graf = data['graf']
    gfs_pwat = data['gfs_pwat']
    gfs_r = data['gfs_r']
    gfs_cape = data['gfs_cape']

    graf_max = np.max(graf, axis=(1,2))

    # Define categories
    dry = graf_max < 0.01
    light = (graf_max >= 0.01) & (graf_max < 2.5)
    moderate = (graf_max >= 2.5) & (graf_max < 10.0)
    heavy = graf_max >= 10.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PWAT
    if np.sum(dry) > 0:
        axes[0].hist(gfs_pwat[dry].flatten(), bins=50, alpha=0.5, label='Dry', color='brown')
    if np.sum(light) > 0:
        axes[0].hist(gfs_pwat[light].flatten(), bins=50, alpha=0.5, label='Light', color='blue')
    if np.sum(moderate) > 0:
        axes[0].hist(gfs_pwat[moderate].flatten(), bins=50, alpha=0.5, label='Moderate', color='green')
    if np.sum(heavy) > 0:
        axes[0].hist(gfs_pwat[heavy].flatten(), bins=50, alpha=0.5, label='Heavy', color='red')
    axes[0].set_xlabel('PWAT (kg/m²)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('GFS PWAT Distribution by Precip Category')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RH
    if np.sum(dry) > 0:
        axes[1].hist(gfs_r[dry].flatten(), bins=50, alpha=0.5, label='Dry', color='brown')
    if np.sum(light) > 0:
        axes[1].hist(gfs_r[light].flatten(), bins=50, alpha=0.5, label='Light', color='blue')
    if np.sum(moderate) > 0:
        axes[1].hist(gfs_r[moderate].flatten(), bins=50, alpha=0.5, label='Moderate', color='green')
    if np.sum(heavy) > 0:
        axes[1].hist(gfs_r[heavy].flatten(), bins=50, alpha=0.5, label='Heavy', color='red')
    axes[1].set_xlabel('Column RH (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('GFS RH Distribution by Precip Category')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CAPE (log scale due to long tail)
    if np.sum(dry) > 0:
        axes[2].hist(np.log10(gfs_cape[dry].flatten() + 1), bins=50, alpha=0.5, label='Dry', color='brown')
    if np.sum(light) > 0:
        axes[2].hist(np.log10(gfs_cape[light].flatten() + 1), bins=50, alpha=0.5, label='Light', color='blue')
    if np.sum(moderate) > 0:
        axes[2].hist(np.log10(gfs_cape[moderate].flatten() + 1), bins=50, alpha=0.5, label='Moderate', color='green')
    if np.sum(heavy) > 0:
        axes[2].hist(np.log10(gfs_cape[heavy].flatten() + 1), bins=50, alpha=0.5, label='Heavy', color='red')
    axes[2].set_xlabel('log10(CAPE + 1) [J/kg]')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('GFS CAPE Distribution by Precip Category')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python diagnose_gfs_bias.py <YYYYMMDDHH> <lead>")
        print("Example: python diagnose_gfs_bias.py 2025120100 12")
        sys.exit(1)

    date = sys.argv[1]
    lead = sys.argv[2]

    print(f"\nLoading training data for {date} at {lead}h lead time...")
    data = load_training_data(date, lead)

    if data is None:
        sys.exit(1)

    analyze_dry_wet_distribution(data)
    create_gfs_distribution_plot(data,
                                  output_file=f'../resnet_data/plots/gfs_distribution_{date}_{lead}h.png')

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nKey Findings Summary:")
    print("1. Check if dry patch percentage is < 10% → sampling bias confirmed")
    print("2. Check if GFS overlap percentages are > 20% → feature ambiguity confirmed")
    print("3. Review the distribution plot for feature separability")
    print("\nSee: GFS_wet_bias_analysis.md for detailed hypotheses and solutions")
