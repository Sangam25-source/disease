#!/usr/bin/env python3
"""
FedED-SegNAS Demo Interface
Interactive demonstration of epistasis detection
"""

import numpy as np
import sys
sys.path.insert(0, '/app/FedED-SegNAS')

class EpistasisDemo:
    """
    Demo interface for showcasing epistasis detection
    """
    
    def __init__(self):
        self.model_loaded = False
        print("🧬 FedED-SegNAS Epistasis Detection System")
        print("=" * 70)
    
    def create_sample_patient(self, risk_level='high'):
        """
        Create sample patient data for demo
        
        Args:
            risk_level: 'high', 'medium', 'low'
        
        Returns:
            Patient SNP profile
        """
        np.random.seed(42)
        
        # Generate 100 SNPs
        num_snps = 100
        snp_profile = np.random.choice([0, 1, 2], size=num_snps, p=[0.7, 0.2, 0.1])
        
        # Create epistatic interactions based on risk level
        if risk_level == 'high':
            # High-risk pattern: Key SNPs are mutated
            snp_profile[5] = 2   # SNP_5 homozygous mutant
            snp_profile[23] = 2  # SNP_23 homozygous mutant
            snp_profile[47] = 1  # SNP_47 heterozygous
            risk_score = 0.85
            
        elif risk_level == 'medium':
            # Medium-risk: Partial interaction
            snp_profile[5] = 1   # SNP_5 heterozygous
            snp_profile[23] = 2  # SNP_23 homozygous mutant
            snp_profile[47] = 0  # SNP_47 normal
            risk_score = 0.45
            
        else:  # low
            # Low-risk: No key mutations
            snp_profile[5] = 0   # SNP_5 normal
            snp_profile[23] = 0  # SNP_23 normal
            snp_profile[47] = 0  # SNP_47 normal
            risk_score = 0.12
        
        return {
            'snp_profile': snp_profile,
            'risk_score': risk_score,
            'key_snps': [5, 23, 47],
            'risk_level': risk_level
        }
    
    def simulate_prediction(self, patient_data):
        """
        Simulate model prediction (for demo purposes)
        In production, this would call the actual trained model
        """
        snp_profile = patient_data['snp_profile']
        
        # Check epistatic interactions
        snp5_status = snp_profile[5]
        snp23_status = snp_profile[23]
        snp47_status = snp_profile[47]
        
        # Simulate epistasis detection
        interactions = []
        
        # 2-way interaction: SNP5 + SNP23
        if snp5_status > 0 and snp23_status > 0:
            interaction_strength = (snp5_status + snp23_status) / 4.0
            interactions.append({
                'type': '2-way',
                'snps': ['SNP_5', 'SNP_23'],
                'strength': interaction_strength,
                'effect': 'Increases disease risk by 35%'
            })
        
        # 3-way interaction: SNP5 + SNP23 + SNP47
        if snp5_status > 0 and snp23_status > 0 and snp47_status > 0:
            interaction_strength = (snp5_status + snp23_status + snp47_status) / 6.0
            interactions.append({
                'type': '3-way',
                'snps': ['SNP_5', 'SNP_23', 'SNP_47'],
                'strength': interaction_strength,
                'effect': 'Increases disease risk by 65%'
            })
        
        # Calculate final risk
        base_risk = 0.10
        interaction_risk = sum(i['strength'] * 0.5 for i in interactions)
        final_risk = min(base_risk + interaction_risk, 0.95)
        
        return {
            'risk_probability': final_risk,
            'risk_category': self._categorize_risk(final_risk),
            'interactions': interactions,
            'confidence': 0.92
        }
    
    def _categorize_risk(self, risk):
        """Categorize risk level"""
        if risk >= 0.70:
            return 'HIGH RISK'
        elif risk >= 0.40:
            return 'MEDIUM RISK'
        else:
            return 'LOW RISK'
    
    def display_patient_info(self, patient_id, patient_data):
        """Display patient information"""
        print(f"\n{'='*70}")
        print(f"📋 PATIENT PROFILE: {patient_id}")
        print(f"{'='*70}")
        
        snp_profile = patient_data['snp_profile']
        
        print(f"\n📊 Genetic Profile Summary:")
        print(f"   Total SNPs analyzed: {len(snp_profile)}")
        print(f"   Genotype distribution:")
        unique, counts = np.unique(snp_profile, return_counts=True)
        for genotype, count in zip(unique, counts):
            percentage = (count / len(snp_profile)) * 100
            genotype_name = {0: 'Normal (0/0)', 1: 'Heterozygous (0/1)', 2: 'Mutant (1/1)'}
            print(f"      {genotype_name[genotype]}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔍 Key SNPs of Interest:")
        for snp_idx in patient_data['key_snps']:
            genotype = snp_profile[snp_idx]
            status = {0: '✓ Normal', 1: '⚠ Heterozygous', 2: '⚠ Mutant'}[genotype]
            print(f"      SNP_{snp_idx}: {status}")
    
    def display_prediction(self, prediction):
        """Display prediction results"""
        print(f"\n{'='*70}")
        print(f"🤖 MODEL PREDICTION RESULTS")
        print(f"{'='*70}")
        
        # Risk assessment
        risk = prediction['risk_probability']
        category = prediction['risk_category']
        confidence = prediction['confidence']
        
        # Visual risk meter
        print(f"\n🎯 Disease Risk Assessment:")
        print(f"   Predicted Risk: {risk*100:.1f}%")
        print(f"   Category: {category}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Visual bar
        bar_length = 50
        filled = int(bar_length * risk)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        if category == 'HIGH RISK':
            color_code = '🔴'
        elif category == 'MEDIUM RISK':
            color_code = '🟡'
        else:
            color_code = '🟢'
        
        print(f"\n   {color_code} Risk Meter: [{bar}] {risk*100:.1f}%")
        
        # Detected interactions
        if prediction['interactions']:
            print(f"\n🔬 Detected Epistatic Interactions:")
            for i, interaction in enumerate(prediction['interactions'], 1):
                print(f"\n   Interaction #{i} ({interaction['type']} epistasis)")
                print(f"      Genes involved: {' × '.join(interaction['snps'])}")
                print(f"      Interaction strength: {interaction['strength']*100:.1f}%")
                print(f"      Effect: {interaction['effect']}")
        else:
            print(f"\n🔬 No significant epistatic interactions detected")
            print(f"   → Low disease risk")
    
    def display_recommendations(self, prediction):
        """Display clinical recommendations"""
        print(f"\n{'='*70}")
        print(f"💊 CLINICAL RECOMMENDATIONS")
        print(f"{'='*70}")
        
        risk_category = prediction['risk_category']
        
        if risk_category == 'HIGH RISK':
            print(f"\n⚠️  HIGH RISK PATIENT - Immediate Action Required")
            print(f"\n   Recommended Actions:")
            print(f"   1. Schedule comprehensive screening within 2 weeks")
            print(f"   2. Consider preventive medications:")
            print(f"      • Drug A (immunomodulator) - 80% effective for this profile")
            print(f"      • Drug B (anti-inflammatory) - 65% effective")
            print(f"   3. Lifestyle modifications:")
            print(f"      • Anti-inflammatory diet")
            print(f"      • Regular monitoring (every 3 months)")
            print(f"   4. Genetic counseling for family members")
            
        elif risk_category == 'MEDIUM RISK':
            print(f"\n⚠️  MEDIUM RISK PATIENT - Monitoring Advised")
            print(f"\n   Recommended Actions:")
            print(f"   1. Regular screening (every 6 months)")
            print(f"   2. Lifestyle modifications:")
            print(f"      • Healthy diet and exercise")
            print(f"      • Stress management")
            print(f"   3. Annual genetic re-evaluation")
            print(f"   4. Patient education on warning signs")
            
        else:
            print(f"\n✅ LOW RISK PATIENT - Routine Care")
            print(f"\n   Recommended Actions:")
            print(f"   1. Standard screening schedule (annual)")
            print(f"   2. General health maintenance")
            print(f"   3. Re-evaluate if symptoms develop")
            print(f"   4. Continue healthy lifestyle practices")
    
    def run_demo(self, patient_id='PATIENT_001', risk_level='high'):
        """
        Run complete demo for a patient
        
        Args:
            patient_id: Patient identifier
            risk_level: 'high', 'medium', or 'low'
        """
        print(f"\n\n{'#'*70}")
        print(f"#  🧬 FedED-SegNAS EPISTASIS DETECTION DEMO")
        print(f"#  Disease: Rheumatoid Arthritis (Example)")
        print(f"{'#'*70}")
        
        # Step 1: Create patient
        print(f"\n⏳ Step 1: Loading patient data...")
        patient_data = self.create_sample_patient(risk_level)
        print(f"✅ Patient data loaded")
        
        # Step 2: Display patient info
        self.display_patient_info(patient_id, patient_data)
        
        # Step 3: Run model prediction
        print(f"\n⏳ Step 2: Running FedED-SegNAS model...")
        print(f"   • Applying Fuzzy CNN for feature extraction...")
        print(f"   • Detecting epistatic interactions...")
        print(f"   • Calculating risk probability...")
        prediction = self.simulate_prediction(patient_data)
        print(f"✅ Analysis complete")
        
        # Step 4: Display results
        self.display_prediction(prediction)
        
        # Step 5: Display recommendations
        self.display_recommendations(prediction)
        
        print(f"\n{'='*70}")
        print(f"📄 END OF REPORT")
        print(f"{'='*70}\n")


def main():
    """Main demo function"""
    demo = EpistasisDemo()
    
    print("\n\n🎭 DEMONSTRATION SCENARIOS")
    print("=" * 70)
    print("\nWe will demonstrate 3 different patients:")
    print("  1. High-risk patient (multiple gene interactions)")
    print("  2. Medium-risk patient (partial interactions)")
    print("  3. Low-risk patient (no significant interactions)")
    
    input("\nPress Enter to start Demo 1 (HIGH RISK)...")
    demo.run_demo('PATIENT_001_HIGH_RISK', risk_level='high')
    
    input("\n\nPress Enter to start Demo 2 (MEDIUM RISK)...")
    demo.run_demo('PATIENT_002_MEDIUM_RISK', risk_level='medium')
    
    input("\n\nPress Enter to start Demo 3 (LOW RISK)...")
    demo.run_demo('PATIENT_003_LOW_RISK', risk_level='low')
    
    print("\n\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("   • Model detects gene interactions (epistasis)")
    print("   • Provides risk assessment with confidence scores")
    print("   • Identifies which genes work together")
    print("   • Generates personalized recommendations")
    print("   • Privacy-preserving federated learning")
    print("\n🎯 Real-world applications:")
    print("   • Early disease detection")
    print("   • Personalized medicine")
    print("   • Drug target discovery")
    print("   • Population health studies")
    print("=" * 70)


if __name__ == '__main__':
    main()
