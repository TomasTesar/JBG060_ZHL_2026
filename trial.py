from processing_data.loading_impact_data import load_admin_boundaries, load_ipc_data

admin1, admin2 = load_admin_boundaries()
ipc = load_ipc_data()

county_cols = set(c for c in ipc.columns if c not in ("Start Date", "End Date"))
admin2_names = set(admin2["adm2_name"].str.lower())

extra_in_ipc = {c for c in county_cols if c.lower() not in admin2_names}
print("In IPC but not in admin2:", extra_in_ipc)

missing_from_ipc = {c for c in admin2_names if c not in {x.lower() for x in county_cols}}
print("In admin2 but not in IPC:", missing_from_ipc)