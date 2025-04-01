"""SQL script generators for implementing BigQuery optimization recommendations."""

from typing import Dict, List, Any
import os

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ScriptGenerator:
    """Generates SQL scripts for implementing recommendations."""
    
    def __init__(self, implementation_plan: Dict[str, Any]):
        """Initialize the script generator.
        
        Args:
            implementation_plan: The implementation plan from the planner
        """
        self.plan = implementation_plan
        
    def generate_scripts(self, output_dir: str) -> Dict[str, str]:
        """Generate SQL scripts for all recommendations.
        
        Args:
            output_dir: Directory to write scripts to
            
        Returns:
            Dict mapping script names to file paths
        """
        script_paths = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate phase scripts
        for i, phase in enumerate(self.plan["phases"]):
            phase_name = phase["name"].replace(":", "").replace(" ", "_").lower()
            phase_file = os.path.join(output_dir, f"{i+1:02d}_{phase_name}.sql")
            
            with open(phase_file, "w") as f:
                f.write(f"-- {phase['name']}\n")
                f.write(f"-- Recommendations: {phase['recommendation_count']}\n")
                f.write(f"-- Estimated Annual Savings: ${phase['estimated_annual_savings_usd']:.2f}\n")
                f.write("\n")
                
                for step in phase["steps"]:
                    f.write(f"-- Step {step['step_number']}: {step['recommendation']}\n")
                    f.write(f"-- Table: {step['table_id']}\n")
                    f.write("\n")
                    
                    for i, impl_step in enumerate(step["implementation_steps"]):
                        f.write(f"-- {impl_step['description']}\n")
                        f.write(f"{impl_step['sql']}\n\n")
                        
                    f.write("-- -----------------------------------------------\n\n")
            
            script_paths[phase_name] = phase_file
            
        # Generate table-specific scripts
        for table_plan in self.plan["table_specific_plans"]:
            table_id = table_plan["table_id"]
            table_file = os.path.join(output_dir, f"table_{table_id}.sql")
            
            with open(table_file, "w") as f:
                f.write(f"-- Optimization Script for Table: {table_id}\n")
                f.write(f"-- Size: {table_plan['table_size_gb']:.2f} GB\n")
                f.write(f"-- Recommendations: {table_plan['recommendation_count']}\n")
                f.write(f"-- Estimated Annual Savings: ${table_plan['estimated_annual_savings_usd']:.2f}\n")
                f.write("\n")
                
                for i, step in enumerate(table_plan["implementation_steps"]):
                    f.write(f"-- Step {i+1}: {step['description']}\n")
                    f.write(f"{step['sql']}\n\n")
            
            script_paths[f"table_{table_id}"] = table_file
            
        # Generate a master script
        master_file = os.path.join(output_dir, "00_master_script.sql")
        with open(master_file, "w") as f:
            f.write(f"-- Master Implementation Script for {self.plan['dataset_id']}\n")
            f.write(f"-- Total Recommendations: {self.plan['total_recommendations']}\n")
            f.write(f"-- Estimated Annual Savings: ${self.plan['roi_summary']['total_annual_savings_usd']:.2f}\n")
            f.write(f"-- Implementation Cost: ${self.plan['roi_summary']['total_implementation_cost_usd']:.2f}\n")
            f.write(f"-- ROI: {self.plan['roi_summary']['overall_roi']:.2f}\n")
            f.write(f"-- Payback Period: {self.plan['roi_summary']['payback_period_months']:.1f} months\n")
            f.write("\n")
            f.write("-- This is a master script that includes all recommendations.\n")
            f.write("-- It's recommended to implement changes phase by phase and test after each implementation.\n")
            f.write("\n")
            
            # Include common utility functions
            f.write("-- Utility function to compare table row counts\n")
            f.write("CREATE OR REPLACE FUNCTION temp.compare_table_counts(original_table STRING, new_table STRING)\n")
            f.write("RETURNS STRUCT<original_count INT64, new_count INT64, match BOOL>\n")
            f.write("AS ((\n")
            f.write("  WITH counts AS (\n")
            f.write("    SELECT\n")
            f.write("      (SELECT COUNT(*) FROM `${original_table}`) AS original_count,\n")
            f.write("      (SELECT COUNT(*) FROM `${new_table}`) AS new_count\n")
            f.write("  )\n")
            f.write("  SELECT\n")
            f.write("    original_count,\n")
            f.write("    new_count,\n")
            f.write("    original_count = new_count AS match\n")
            f.write("  FROM counts\n")
            f.write("));\n\n")
            
            # Include reference to each phase script
            f.write("-- Implementation Phases\n")
            for i, phase in enumerate(self.plan["phases"]):
                phase_name = phase["name"].replace(":", "").replace(" ", "_").lower()
                f.write(f"-- Phase {i+1}: {phase['name']}\n")
                f.write(f"-- See: {i+1:02d}_{phase_name}.sql\n\n")
                
        script_paths["master_script"] = master_file
        
        return script_paths
                
    def generate_rollback_script(self, output_dir: str) -> str:
        """Generate rollback script to undo changes.
        
        Args:
            output_dir: Directory to write script to
            
        Returns:
            Path to the rollback script
        """
        rollback_file = os.path.join(output_dir, "rollback.sql")
        
        with open(rollback_file, "w") as f:
            f.write("-- Rollback Script\n")
            f.write("-- Use this script to revert the changes made by the optimization scripts\n\n")
            
            # Get all tables being modified
            tables = set()
            for phase in self.plan["phases"]:
                for step in phase["steps"]:
                    tables.add(step["table_id"])
            
            # Generate rollback statements for each table
            project_id = self.plan["project_id"]
            dataset_id = self.plan["dataset_id"]
            
            for table in tables:
                f.write(f"-- Rollback changes to {table}\n")
                f.write(f"BEGIN\n")
                f.write(f"  -- Check if backup exists\n")
                f.write(f"  IF EXISTS(SELECT 1 FROM `{project_id}.{dataset_id}.__INFORMATION_SCHEMA__.__TABLES__` WHERE table_name = '{table}_backup') THEN\n")
                f.write(f"    -- Restore from backup\n")
                f.write(f"    DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    ALTER TABLE `{project_id}.{dataset_id}.{table}` RENAME TO `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    ALTER TABLE `{project_id}.{dataset_id}.{table}_backup` RENAME TO `{project_id}.{dataset_id}.{table}`;\n")
                f.write(f"    DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    PRINT 'Rolled back {table} from backup';\n")
                f.write(f"  ELSEIF EXISTS(SELECT 1 FROM `{project_id}.{dataset_id}.__INFORMATION_SCHEMA__.__TABLES__` WHERE table_name = '{table}_old') THEN\n")
                f.write(f"    -- Restore from _old\n")
                f.write(f"    DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    ALTER TABLE `{project_id}.{dataset_id}.{table}` RENAME TO `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    ALTER TABLE `{project_id}.{dataset_id}.{table}_old` RENAME TO `{project_id}.{dataset_id}.{table}`;\n")
                f.write(f"    DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table}_temp`;\n")
                f.write(f"    PRINT 'Rolled back {table} from _old table';\n")
                f.write(f"  ELSE\n")
                f.write(f"    PRINT 'No backup found for {table}';\n")
                f.write(f"  END IF;\n")
                f.write(f"END;\n\n")
                
            # Clean up any temporary tables
            f.write("-- Clean up remaining backup and temporary tables\n")
            f.write(f"BEGIN\n")
            f.write(f"  DECLARE backup_tables ARRAY<STRING>;\n")
            f.write(f"  SET backup_tables = (\n")
            f.write(f"    SELECT ARRAY_AGG(table_name)\n")
            f.write(f"    FROM `{project_id}.{dataset_id}.__INFORMATION_SCHEMA__.__TABLES__`\n")
            f.write(f"    WHERE table_name LIKE '%_backup' OR table_name LIKE '%_old' OR table_name LIKE '%_temp'\n")
            f.write(f"      OR table_name LIKE '%_optimized' OR table_name LIKE '%_partitioned' OR table_name LIKE '%_clustered'\n")
            f.write(f"  );\n\n")
            f.write(f"  FOR table_name IN (SELECT * FROM UNNEST(backup_tables))\n")
            f.write(f"  DO\n")
            f.write(f"    EXECUTE IMMEDIATE 'DROP TABLE IF EXISTS `{project_id}.{dataset_id}.' || table_name || '`';\n")
            f.write(f"  END FOR;\n")
            f.write(f"END;\n")
            
        return rollback_file
