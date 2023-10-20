using UnityEngine;
using System.Collections;
using System.Linq;
using System.Security.Cryptography.X509Certificates;

public class wave_motion : MonoBehaviour
{
	int size = 100;
	float rate = 0.005f;
	float gamma = 0.004f;
	float damping = 0.996f;
	float[,] old_h;
	float[,] low_h;
	float[,] vh;
	float[,] b;

	bool[,] cg_mask;
	float[,] cg_p;
	float[,] cg_r;
	float[,] cg_Ap;
	bool tag = true;

	Vector3 cube_v = Vector3.zero;
	Vector3 cube_w = Vector3.zero;

	GameObject cube;
	Bounds cube_bbox;
	GameObject block;
	Bounds block_bbox;

	// Use this for initialization
	void Start()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		mesh.Clear();

		Vector3[] X = new Vector3[size * size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				X[i * size + j].x = i * 0.1f - size * 0.05f;
				X[i * size + j].y = 0;
				X[i * size + j].z = j * 0.1f - size * 0.05f;
			}

		int[] T = new int[(size - 1) * (size - 1) * 6];
		int index = 0;
		for (int i = 0; i < size - 1; i++)
			for (int j = 0; j < size - 1; j++)
			{
				T[index * 6 + 0] = (i + 0) * size + (j + 0);
				T[index * 6 + 1] = (i + 0) * size + (j + 1);
				T[index * 6 + 2] = (i + 1) * size + (j + 1);
				T[index * 6 + 3] = (i + 0) * size + (j + 0);
				T[index * 6 + 4] = (i + 1) * size + (j + 1);
				T[index * 6 + 5] = (i + 1) * size + (j + 0);
				index++;
			}
		mesh.vertices = X;
		mesh.triangles = T;
		mesh.RecalculateNormals();

		low_h = new float[size, size];
		old_h = new float[size, size];
		vh = new float[size, size];
		b = new float[size, size];

		cg_mask = new bool[size, size];
		cg_p = new float[size, size];
		cg_r = new float[size, size];
		cg_Ap = new float[size, size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				low_h[i, j] = 99999;
				old_h[i, j] = 0;
				vh[i, j] = 0;
			}

		cube = GameObject.Find("Cube");
		block = GameObject.Find("Block");
		// Mesh.bounds and localBounds are similar but they return the bounds in local space.
		// https://docs.unity3d.com/ScriptReference/Renderer-bounds.html
		cube_bbox = cube.GetComponent<MeshFilter>().mesh.bounds;
		block_bbox = block.GetComponent<MeshFilter>().mesh.bounds;
	}

	void A_Times(bool[,] mask, float[,] x, float[,] Ax, int li, int ui, int lj, int uj)
	{
		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					Ax[i, j] = 0;
					if (i != 0) Ax[i, j] -= x[i - 1, j] - x[i, j];
					if (i != size - 1) Ax[i, j] -= x[i + 1, j] - x[i, j];
					if (j != 0) Ax[i, j] -= x[i, j - 1] - x[i, j];
					if (j != size - 1) Ax[i, j] -= x[i, j + 1] - x[i, j];
				}
	}

	float Dot(bool[,] mask, float[,] x, float[,] y, int li, int ui, int lj, int uj)
	{
		float ret = 0;
		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					ret += x[i, j] * y[i, j];
				}
		return ret;
	}

	void Conjugate_Gradient(bool[,] mask, float[,] b, float[,] x, int li, int ui, int lj, int uj)
	{
		//Solve the Laplacian problem by CG.
		A_Times(mask, x, cg_r, li, ui, lj, uj);

		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					cg_p[i, j] = cg_r[i, j] = b[i, j] - cg_r[i, j];
				}

		float rk_norm = Dot(mask, cg_r, cg_r, li, ui, lj, uj);

		for (int k = 0; k < 128; k++)
		{
			if (rk_norm < 1e-10f) break;
			A_Times(mask, cg_p, cg_Ap, li, ui, lj, uj);
			float alpha = rk_norm / Dot(mask, cg_p, cg_Ap, li, ui, lj, uj);

			for (int i = li; i <= ui; i++)
				for (int j = lj; j <= uj; j++)
					if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
					{
						x[i, j] += alpha * cg_p[i, j];
						cg_r[i, j] -= alpha * cg_Ap[i, j];
					}

			float _rk_norm = Dot(mask, cg_r, cg_r, li, ui, lj, uj);
			float beta = _rk_norm / rk_norm;
			rk_norm = _rk_norm;

			for (int i = li; i <= ui; i++)
				for (int j = lj; j <= uj; j++)
					if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
					{
						cg_p[i, j] = cg_r[i, j] + beta * cg_p[i, j];
					}
		}

	}

	bool in_zone(int i, int j)
	{
		return i >= 0 && i < size && j >= 0 && j < size;
	}
	void Shallow_Wave(float[,] old_h, float[,] h, float[,] new_h)
	{
		//Step 1:
		//TODO: Compute new_h based on the shallow wave model.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				new_h[i, j] = h[i, j] + damping * (h[i, j] - old_h[i, j]);
				if (in_zone(i - 1, j)) new_h[i, j] += rate * (h[i - 1, j] - h[i, j]);
				if (in_zone(i + 1, j)) new_h[i, j] += rate * (h[i + 1, j] - h[i, j]);
				if (in_zone(i, j - 1)) new_h[i, j] += rate * (h[i, j - 1] - h[i, j]);
				if (in_zone(i, j + 1)) new_h[i, j] += rate * (h[i, j + 1] - h[i, j]);
			}

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				low_h[i, j] = 99999;
			}

		//Step 2: Block->Water coupling
		//TODO: for block 1, calculate low_h.
		//TODO: then set up b and cg_mask for conjugate gradient.
		//TODO: Solve the Poisson equation to obtain vh (virtual height).
		Vector3[] X = GetComponent<MeshFilter>().mesh.vertices;
		float expand_ = 1.73f * 0.5f;
		Vector3 block_pos = block.transform.position;
		int block_min_i = (int)((block_pos.x - expand_ + size * 0.05f) / 0.1);
		int block_max_i = (int)((block_pos.x + expand_ + size * 0.05f) / 0.1);
		int block_min_j = (int)((block_pos.z - expand_ + size * 0.05f) / 0.1);
		int block_max_j = (int)((block_pos.z + expand_ + size * 0.05f) / 0.1);

		for (int i = block_min_i; i <= block_max_i; i++)
			for (int j = block_min_j; j <= block_max_j; j++)
			{
				if (!in_zone(i, j)) continue;
				Vector3 local_ray_o = block.transform.InverseTransformPoint(new Vector3(X[i * size + j].x, -8, X[i * size + j].z));
				Vector3 local_ray_p = block.transform.InverseTransformPoint(new Vector3(X[i * size + j].x, -7, X[i * size + j].z));
				Ray ray = new Ray(local_ray_o, local_ray_p - local_ray_o);
				float dist = 99999;
				block_bbox.IntersectRay(ray, out dist);
				low_h[i, j] = -8 + dist;
			}
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (low_h[i, j] < new_h[i, j])
				{
					b[i, j] = (new_h[i, j] - low_h[i, j]) / rate;
					cg_mask[i, j] = true;
				}
				else
				{
					b[i, j] = 0;
					vh[i, j] = 0;
					cg_mask[i, j] = false;
				}
			}


		Conjugate_Gradient(cg_mask, b, vh, block_min_i - 1, block_max_i + 1, block_min_j - 1, block_max_j + 1);

		//TODO: for block 2, calculate low_h.
		//TODO: then set up b and cg_mask for conjugate gradient.
		//TODO: Solve the Poisson equation to obtain vh (virtual height).

		Vector3 cube_pos = cube.transform.position;
		int cube_min_i = (int)((cube_pos.x - expand_ + size * 0.05f) / 0.1);
		int cube_max_i = (int)((cube_pos.x + expand_ + size * 0.05f) / 0.1);
		int cube_min_j = (int)((cube_pos.z - expand_ + size * 0.05f) / 0.1);
		int cube_max_j = (int)((cube_pos.z + expand_ + size * 0.05f) / 0.1);

		for (int i = cube_min_i; i <= cube_max_i; i++)
			for (int j = cube_min_j; j <= cube_max_j; j++)
			{
				if (!in_zone(i, j)) continue;
				Vector3 local_ray_o = cube.transform.InverseTransformPoint(new Vector3(X[i * size + j].x, -8, X[i * size + j].z));
				Vector3 local_ray_p = cube.transform.InverseTransformPoint(new Vector3(X[i * size + j].x, -7, X[i * size + j].z));
				Ray ray = new Ray(local_ray_o, local_ray_p - local_ray_o);
				float dist = 99999;
				cube_bbox.IntersectRay(ray, out dist);
				low_h[i, j] = -8 + dist;
			}
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (low_h[i, j] < new_h[i, j])
				{
					b[i, j] = (new_h[i, j] - low_h[i, j]) / rate;
					cg_mask[i, j] = true;
				}
				else
				{
					b[i, j] = 0;
					vh[i, j] = 0;
					cg_mask[i, j] = false;
				}
			}


		Conjugate_Gradient(cg_mask, b, vh, cube_min_i - 1, cube_max_i + 1, cube_min_j - 1, cube_max_j + 1);

		//TODO: Diminish vh.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				if (cg_mask[i, j])
					vh[i, j] *= gamma;

		//TODO: Update new_h by vh.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (i != 0)
					new_h[i, j] += rate * (vh[i - 1, j] - vh[i, j]);
				if (i != size - 1)
					new_h[i, j] += rate * (vh[i + 1, j] - vh[i, j]);
				if (j != 0)
					new_h[i, j] += rate * (vh[i, j - 1] - vh[i, j]);
				if (j != size - 1)
					new_h[i, j] += rate * (vh[i, j + 1] - vh[i, j]);
			}


		//Step 3
		//TODO: old_h <- h; h <- new_h;
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				old_h[i, j] = h[i, j];
				h[i, j] = new_h[i, j];
			}

		//Step 4: Water->Block coupling.
		//More TODO here.
		Quaternion q = cube.transform.rotation;
		Matrix4x4 R = Matrix4x4.Rotate(q);
		float t = 0.004f;
		float Mass = 333.0f;
		float rho_water = 997;
		Vector3 cube_net_force = Mass * new Vector3(0, -9.8f, 0);
		Vector3 cube_net_torque = new Vector3(0, 0, 0);
		for (int i = cube_min_i; i <= cube_max_i; i++)
			for (int j = cube_min_j; j <= cube_max_j; j++)
			{
				if ((!in_zone(i, j)) || vh[i, j] <= 0) continue;
				Vector3 buoyancy = new Vector3(0, 9.8f, 0) * rho_water * (0.1f * 0.1f * vh[i, j]);
				cube_net_force += buoyancy;
				cube_net_torque += Vector3.Cross(
					new Vector3(X[i * size + j].x, low_h[i, j], X[i * size + j].z) - cube_pos,
					buoyancy);
			}

		cube_v *= 0.99f;
		cube_w *= 0.99f;
		cube_v += cube_net_force * t / Mass;
		cube.transform.position += cube_v * t;
		cube_w += cube_net_torque * t / (1.0f * Mass);
		Quaternion cube_q = cube.transform.rotation;
		Quaternion wq = new Quaternion(cube_w.x, cube_w.y, cube_w.z, 0);
		Quaternion temp_q = wq * cube_q;
		cube_q.x += 0.5f * t * temp_q.x;
		cube_q.y += 0.5f * t * temp_q.y;
		cube_q.z += 0.5f * t * temp_q.z;
		cube_q.w += 0.5f * t * temp_q.w;
		cube.transform.rotation = cube_q;

	}


	// Update is called once per frame
	void Update()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] X = mesh.vertices;
		float[,] new_h = new float[size, size];
		float[,] h = new float[size, size];

		//TODO: Load X.y into h.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				h[i, j] = X[i * size + j].y;
			}

		if (Input.GetKeyDown("r"))
		{
			//TODO: Add random water.
			int i = Random.Range(0, size);
			int j = Random.Range(0, size);
			int valid_neighbor_num = 0;
			float r = Random.Range(1f, 3f);
			for (int ii = i - 1; ii <= i + 1; ii++)
				for (int jj = j - 1; jj <= j + 1; jj++)
				{
					if (ii == i && jj == j) continue;
					if (in_zone(ii, jj)) valid_neighbor_num++;
				}
			h[i, j] += r;
			for (int ii = i - 1; ii <= i + 1; ii++)
				for (int jj = j - 1; jj <= j + 1; jj++)
				{
					if (ii == i && jj == j) continue;
					if (ii < 0 || ii >= size || jj < 0 || jj >= size) continue;
					h[ii, jj] -= r / valid_neighbor_num;
				}
		}

		for (int l = 0; l < 8; l++)
		{
			Shallow_Wave(old_h, h, new_h);
		}

		//TODO: Store h back into X.y and recalculate normal.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				X[i * size + j].y = h[i, j];
			}
		mesh.vertices = X;
		mesh.RecalculateNormals();

	}
}
