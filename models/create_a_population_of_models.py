"""
This file is used to create a batch of De Leva models within the range of anthropometry studied.
These models will be used for the predictive simulations.
"""

from pathlib import Path
import csv
import numpy as np
import itertools
from biobuddy import (
    DeLevaTable,
    Sex,
    SegmentReal,
    SegmentCoordinateSystemReal,
    MeshReal,
    MarkerReal,
    RotoTransMatrix,
    MergeSegmentsTool,
    SegmentMerge,
    ModifyKinematicChainTool,
    ChangeFirstSegment,
    Translations,
    Rotations,
    RangeOfMotion,
    Ranges,
)



def create_hand_root_model(
    this_height,
    this_ankle_height,
    this_knee_height,
    this_hip_height,
    this_shoulder_height,
    this_finger_span,
    this_wrist_span,
    this_elbow_span,
    this_shoulder_span,
    this_hip_width,
    this_foot_length,
    this_umbilicus_height,
    this_xiphoid_height,
    this_mass,
):

    # Create the inertial table for this model
    inertia_table = DeLevaTable(this_mass, sex=Sex.FEMALE)
    inertia_table.from_measurements(
        total_height=this_height,
        ankle_height=this_ankle_height,
        knee_height=this_knee_height,
        hip_height=this_hip_height,
        shoulder_height=this_shoulder_height,
        finger_span=this_finger_span,
        wrist_span=this_wrist_span,
        elbow_span=this_elbow_span,
        shoulder_span=this_shoulder_span,
        hip_width=this_hip_width,
        foot_length=this_foot_length,
        umbilicus_height=this_umbilicus_height,
        xiphoid_height=this_xiphoid_height,
    )

    # Create the model
    real_model = inertia_table.to_simple_model()
    # real_model.animate()

    # Modify the model to merge both arms together
    merge_tool = MergeSegmentsTool(real_model)
    merge_tool.add(SegmentMerge(name="UPPER_ARMS", first_segment_name="L_UPPER_ARM", second_segment_name="R_UPPER_ARM"))
    merge_tool.add(SegmentMerge(name="LOWER_ARMS", first_segment_name="L_LOWER_ARM", second_segment_name="R_LOWER_ARM"))
    merge_tool.add(SegmentMerge(name="HANDS", first_segment_name="L_HAND", second_segment_name="R_HAND"))
    merged_model = merge_tool.merge()
    # merged_model.animate()

    # Modify the model to place the root segment at the hands
    kinematic_chain_modifier = ModifyKinematicChainTool(merged_model)
    kinematic_chain_modifier.add(ChangeFirstSegment(first_segment_name="HANDS", new_segment_name="PELVIS"))
    hand_root_model = kinematic_chain_modifier.modify()
    # PLace the hands on the bar
    hand_root_model.segments["HANDS"].segment_coordinate_system.scs.translation = np.array([0, 1.2, 0])

    # Modify the degrees of freedom
    hand_root_model.segments["HANDS"].translations = Translations.XZ
    hand_root_model.segments["HANDS"].rotations = Rotations.Y
    hand_root_model.segments["HANDS"].dof_names = [
                                        'HANDS_transX',
                                         'HANDS_transZ',
                                         'HANDS_rotY',
                                                   ]
    hand_root_model.segments["HANDS"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-0.3, -0.3, -2*np.pi]),
        max_bound=np.array([0.3, 0.3, 2*np.pi]),
    )
    hand_root_model.segments["LOWER_ARMS"].rotations = Rotations.NONE
    hand_root_model.segments["LOWER_ARMS"].dof_names = None
    hand_root_model.segments["UPPER_ARMS"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-2*np.pi/9]),
        max_bound=np.array([0]),
    )
    hand_root_model.segments["UPPER_TRUNK"].rotations = Rotations.Y
    hand_root_model.segments["UPPER_TRUNK"].dof_names = ["UPPER_TRUNK_rotY"]
    hand_root_model.segments["UPPER_TRUNK"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-2*np.pi/9]),
        max_bound=np.array([0]),
    )
    hand_root_model.segments["MID_TRUNK"].rotations = Rotations.NONE
    hand_root_model.segments["MID_TRUNK"].dof_names = None
    hand_root_model.segments["LOWER_TRUNK"].rotations = Rotations.Y
    hand_root_model.segments["LOWER_TRUNK"].dof_names = ["LOWER_TRUNK_rotY"]
    hand_root_model.segments["LOWER_TRUNK"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-np.pi/18]),
        max_bound=np.array([np.pi/6]),
    )
    hand_root_model.segments["HEAD"].rotations = Rotations.Y
    hand_root_model.segments["HEAD"].dof_names = ["HEAD_rotY"]
    hand_root_model.segments["HEAD"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-np.pi/3]),
        max_bound=np.array([5*np.pi/18]),
    )
    hand_root_model.segments["R_THIGH"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-2 * np.pi / 9, -np.pi / 3]),
        max_bound=np.array([0, np.pi / 6]),
    )
    hand_root_model.segments["R_SHANK"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([0]),
        max_bound=np.array([5*np.pi/6]),
    )
    hand_root_model.segments["R_FOOT"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-np.pi / 2]),
        max_bound=np.array([0]),
    )
    hand_root_model.segments["L_THIGH"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([0, -np.pi / 3]),
        max_bound=np.array([2*np.pi/9, np.pi / 6]),
    )
    hand_root_model.segments["L_SHANK"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([0]),
        max_bound=np.array([5*np.pi/6]),
    )
    hand_root_model.segments["L_FOOT"].q_ranges = RangeOfMotion(
        range_type=Ranges.Q,
        min_bound=np.array([-np.pi/2]),
        max_bound=np.array([0]),
    )

    # Add asymmetric bars for visualization and constraints
    lower_bar_scs = RotoTransMatrix.from_euler_angles_and_translation(
        angle_sequence="xyz",
        angles=np.array([0, 0, 0]),
        translation=np.array([0, 0, 1.55])
    )
    hand_root_model.add_segment(
        SegmentReal(
            name="LowerBar",
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=lower_bar_scs),
            mesh=MeshReal(positions=np.array([
                [0, 0, -1.55],
                [0, 0, 0],
                [0, 2.4, 0],
                [0, 2.4, -1.55]
            ])),
        )
    )
    hand_root_model.segments["LowerBar"].add_marker(
        MarkerReal(
            name="LowerBarMarker",
            parent_name="LowerBar",
            position=np.array([0, 0, 0])
        )
    )

    upper_bar_scs = RotoTransMatrix.from_euler_angles_and_translation(
        angle_sequence="xyz",
        angles=np.array([0, 0, 0]),
        translation=np.array([0, 1.62, 2.35])
    )
    hand_root_model.add_segment(
        SegmentReal(
            name="UpperBar",
            segment_coordinate_system=SegmentCoordinateSystemReal(scs=upper_bar_scs),
            mesh=MeshReal(positions=np.array([
                [0, 0, -2.35],
                [0, 0, 0],
                [0, 2.4, 0],
                [0, 2.4, -2.35]
            ])),
        )
    )
    hand_root_model.segments["UpperBar"].add_marker(
        MarkerReal(
            name="UpperBarMarker",
            parent_name="UpperBar",
            position=np.array([0, 0, 0])
        )
    )

    # Add markers on the feet for visualization and constraints
    hand_root_model.segments["L_FOOT"].add_marker(
        MarkerReal(
            name="L_TOES",
            parent_name="L_FOOT",
            position=hand_root_model.segments["L_FOOT"].mesh.positions[:, 1]  # End of toes
        )
    )
    hand_root_model.segments["R_FOOT"].add_marker(
        MarkerReal(
            name="R_TOES",
            parent_name="R_FOOT",
            position=hand_root_model.segments["R_FOOT"].mesh.positions[:, 1]  # End of toes
        )
    )

    # Rotate the root segment
    # hand_root_model.animate()

    return hand_root_model


def main():

    current_path = Path(__file__).parent
    model_path = f"{current_path}/biomod_models"

    # Set the range of anthropometry that you want to create
    # TODO: set these as -std, -1/2std, mean, +1/2std, +std
    # when only one value, it's the mean coef
    total_mass = [46.7, 52.8, 61.2, 70.4]  # Kg
    total_height = [1.53, 1.64, 1.75]  # m
    ankle_height = [0.03]
    knee_height_coeff = [0.255, 0.279]
    hip_height_coeff = [0.495, 0.513]
    shoulder_height_coeff = [0.792, 0.817, 0.831]
    shoulder_width_coeff = [0.201]
    elbow_span_coeff = [0.494, 0.539]
    wrist_span_coeff = [ 0.804, 0.835]
    finger_span_coeff = [1.004]
    foot_length_coeff = [0.106]
    hip_width_coeff = [0.191]
    umbilicus_heihgt_coeff = [0.596]
    xiphoid_height_coeff = [0.739]

    # Create csv file to save which model has which coefficient
    with open(f"{model_path}/../model_coefficients.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")

        # Titles
        writer.writerow([
            "ModelNumber",
            "Mass",
            "Height",
            "AnkleHeightCoeff",
            "KneeHeightCoeff",
            "PelvisHeightCoeff",
            "ShoulderHeightCoeff",
            "ShoulderWidthCoeff",
            "ElbowSpanCoeff",
            "WristSpanCoeff",
            "FingerSpanCoeff",
            "FootLengthCoeff",
            "HipWidthCoeff",
            "UmbilicusHeightCoeff",
            "XiphoidHeightCoeff"
        ])

        # Create all combinations using itertools.product
        model_number = 1
        for combination in itertools.product(
            total_mass,
            total_height,
            ankle_height,
            knee_height_coeff,
            hip_height_coeff,
            shoulder_height_coeff,
            shoulder_width_coeff,
            elbow_span_coeff,
            wrist_span_coeff,
            finger_span_coeff,
            foot_length_coeff,
            hip_width_coeff,
            umbilicus_heihgt_coeff,
            xiphoid_height_coeff,
        ):
            (
                this_mass,
                this_height,
                this_ankle_height_coeff,
                this_knee_height_coeff,
                this_hip_height_coeff,
                this_shoulder_height_coeff,
                this_shoulder_span_coeff,
                this_elbow_span_coeff,
                this_wrist_span_coeff,
                this_finger_span_coeff,
                this_foot_length_coeff,
                this_hip_width_coeff,
                this_umbilicus_height_coeff,
                this_xiphoid_height_coeff
            ) = combination

            # Get the measurements for this model
            this_ankle_height = this_ankle_height_coeff * this_height
            this_knee_height = this_knee_height_coeff * this_height
            this_hip_height = this_hip_height_coeff * this_height
            this_shoulder_height = this_shoulder_height_coeff * this_height
            this_shoulder_span = this_shoulder_span_coeff * this_height
            this_elbow_span = this_elbow_span_coeff * this_height
            this_wrist_span = this_wrist_span_coeff * this_height
            this_finger_span = this_finger_span_coeff * this_height
            this_foot_length = this_foot_length_coeff * this_height
            this_hip_width = this_hip_width_coeff * this_height
            this_umbilicus_height = this_umbilicus_height_coeff * this_height
            this_xiphoid_height = this_xiphoid_height_coeff * this_height

            hand_root_model = create_hand_root_model(
                this_height,
                this_ankle_height,
                this_knee_height,
                this_hip_height,
                this_shoulder_height,
                this_finger_span,
                this_wrist_span,
                this_elbow_span,
                this_shoulder_span,
                this_hip_width,
                this_foot_length,
                this_umbilicus_height,
                this_xiphoid_height,
                this_mass,
            )

            # Exporting the output model as a biomod file
            hand_root_model.to_biomod(f"{model_path}/athlete_{model_number:03d}_deleva.bioMod")

            writer.writerow([
                model_number,
                this_mass,
                this_height,
                this_ankle_height_coeff,
                this_knee_height_coeff,
                this_hip_height_coeff,
                this_shoulder_height_coeff,
                this_shoulder_span_coeff,
                this_elbow_span_coeff,
                this_wrist_span_coeff,
                this_finger_span_coeff,
                this_foot_length_coeff,
                this_hip_width_coeff,
                this_umbilicus_height_coeff,
                this_xiphoid_height_coeff,
            ])

            model_number += 1


if __name__ == "__main__":
    main()
