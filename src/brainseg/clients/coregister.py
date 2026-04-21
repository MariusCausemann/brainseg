import argparse
import ants

def coregister_images(img1_path, img2_path, output_path):
    print(f"Loading Fixed Image (img1): {img1_path}")
    fixed_img = ants.image_read(str(img1_path))
    
    print(f"Loading Moving Image (img2): {img2_path}")
    moving_img = ants.image_read(str(img2_path))
    
    print("Running Rigid Registration (this may take a minute)...")
    # 'Rigid' uses mutual information, perfect for cross-modality registration
    registration = ants.registration(
        fixed=fixed_img, 
        moving=moving_img, 
        type_of_transform='Rigid'
    )
    
    # The output is automatically resampled to match img1's resolution and grid
    registered_img2 = registration['warpedmovout']
    
    print(f"Saving co-registered img2 to: {output_path}")
    ants.image_write(registered_img2, str(output_path))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-register img2 to img1 using ANTsPy.")
    parser.add_argument("--img1", required=True, help="Path to the fixed image (target space)")
    parser.add_argument("--img2", required=True, help="Path to the moving image")
    parser.add_argument("--out", required=True, help="Path to save the registered img2")
    
    args = parser.parse_args()
    coregister_images(args.img1, args.img2, args.out)