using UnityEngine;
using System.Collections;
using Unity.VisualScripting;
using Unity.VisualScripting.AssemblyQualifiedNameParser;
using System.Threading;

public class Rigid_Bunny : MonoBehaviour 
{
	bool launched 		= false;
	float dt 			= 0.015f;
	Vector3 v 			= new Vector3(0, 0, 0);	// velocity
	Vector3 w 			= new Vector3(0, 0, 0); // angular velocity
	Vector3 g			= new Vector3(0, -9.8f, 0);

	float mass;									// mass
	Matrix4x4 I_ref;							// reference inertia

	float linear_decay	= 0.99f;				// for velocity decay
	float angular_decay	= 0.98f;				
	float restitution 	= 0.5f;                 // for collision
	float mu			= 0.35f;

	// Use this for initialization
	void Start () 
	{		
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		float m=1;
		mass=0;
		for (int i=0; i<vertices.Length; i++) 
		{
			mass += m;
			float diag=m*vertices[i].sqrMagnitude;
			I_ref[0, 0]+=diag;
			I_ref[1, 1]+=diag;
			I_ref[2, 2]+=diag;
			I_ref[0, 0]-=m*vertices[i][0]*vertices[i][0];
			I_ref[0, 1]-=m*vertices[i][0]*vertices[i][1];
			I_ref[0, 2]-=m*vertices[i][0]*vertices[i][2];
			I_ref[1, 0]-=m*vertices[i][1]*vertices[i][0];
			I_ref[1, 1]-=m*vertices[i][1]*vertices[i][1];
			I_ref[1, 2]-=m*vertices[i][1]*vertices[i][2];
			I_ref[2, 0]-=m*vertices[i][2]*vertices[i][0];
			I_ref[2, 1]-=m*vertices[i][2]*vertices[i][1];
			I_ref[2, 2]-=m*vertices[i][2]*vertices[i][2];
		}
		I_ref [3, 3] = 1;
	}
	
	Matrix4x4 Get_Cross_Matrix(Vector3 a)
	{
		//Get the cross product matrix of vector a
		Matrix4x4 A = Matrix4x4.zero;
		A [0, 0] = 0; 
		A [0, 1] = -a [2]; 
		A [0, 2] = a [1]; 
		A [1, 0] = a [2]; 
		A [1, 1] = 0; 
		A [1, 2] = -a [0]; 
		A [2, 0] = -a [1]; 
		A [2, 1] = a [0]; 
		A [2, 2] = 0; 
		A [3, 3] = 1;
		return A;
	}

    Matrix4x4 cross_matrix(Vector3 r)
    {
		Matrix4x4 res = new Matrix4x4(new Vector4(   0,  r.z, -r.y, 0),
									  new Vector4(-r.z,    0,  r.x, 0),
									  new Vector4( r.y, -r.x,    0, 0),
									  new Vector4(   0,    0,	 0, 1));
        return res;
    }

    Matrix4x4 matrix_sub(Matrix4x4 a, Matrix4x4 b)
    {
        return new Matrix4x4(a.GetColumn(0) - b.GetColumn(0),
                            a.GetColumn(1) - b.GetColumn(1),
                            a.GetColumn(2) - b.GetColumn(2),
                            new Vector4(0, 0, 0, 1));
    }

    // In this function, update v and w by the impulse due to the collision with
    //a plane <P, N>
    void Collision_Impulse(Vector3 P, Vector3 N)
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		Vector3 x = transform.position;
        Quaternion q = transform.rotation;
        Matrix4x4 R = Matrix4x4.Rotate(q);
        Matrix4x4 I = R * I_ref * R.transpose;

		int n_collid = 0;
		Vector3 avg_ri = new Vector3(0, 0, 0);
		for (int i = 0; i < vertices.Length; i++)
		{
			Vector3 ri_rest = vertices[i];
			Vector3 ri = R * ri_rest;
			Vector3 xi = x + ri;
			if (Vector3.Dot(xi - P, N) < 0)
			{
				Vector3 vi = v + Vector3.Cross(w, ri);
				if (Vector3.Dot(vi, N) < 0)
				{
					avg_ri += ri;
					n_collid++;
				}
			}
		}
		if (n_collid == 0) return;
		avg_ri /= n_collid;
		Vector3 avg_vi = v + Vector3.Cross(w, avg_ri);

		Vector3 vn = Vector3.Dot(avg_vi, N) * N;
		Vector3 vt = avg_vi - vn;
		float a = Mathf.Max(1 - mu * (1 + restitution) * vn.magnitude / vt.magnitude, 0);
		Vector3 vn_new = -1 * restitution * vn;
		Vector3 vt_new = a * vt;

		Vector3  vi_new = vn_new + vt_new;

        float tmp_m = 1.0f / mass;
        Matrix4x4 inverse_m = new Matrix4x4(new Vector4(tmp_m, 0, 0, 0),
                                            new Vector4(0, tmp_m, 0, 0),
                                            new Vector4(0, 0, tmp_m, 0),
                                            new Vector4(0, 0, 0, tmp_m));

        Matrix4x4 cross_mat = cross_matrix(avg_ri);
		Matrix4x4 I_inv = I.inverse;
		Matrix4x4 K = matrix_sub(inverse_m, cross_mat * I_inv * cross_mat);
        Vector3 j = K.inverse.MultiplyVector(vi_new - avg_vi);

		v = v + tmp_m * j;
		w = w + I.inverse.MultiplyVector(Vector3.Cross(avg_ri, j));

		restitution *= 0.7f;
		if (v.magnitude < 0.01f || w.magnitude < 0.01f)
			restitution = 0;
	}

	// Update is called once per frame
	void Update () 
	{
		//Game Control
		if(Input.GetKey("r"))
		{
			transform.position = new Vector3 (0, 0.6f, 0);
			restitution = 0.5f;
			launched=false;
		}
		if(Input.GetKey("l"))
		{
			v = new Vector3 (5, 3, 0);
			restitution = 0.5f;
			launched=true;
		}

		if (!launched)
			return;


		// Part I: Update velocities
        v = v + dt * g;

		v = linear_decay * v;
		w = angular_decay * w;

		// Part II: Collision Impulse
		Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0));
		Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));

		// Part III: Update position & orientation
		//Update linear status
		Vector3 x    = transform.position;
		x = x + dt * v;

		//Update angular status
		Quaternion q = transform.rotation;
		Vector3 tmp0 = dt * 0.5f * w;
		Quaternion tmp1 = new Quaternion(tmp0.x, tmp0.y, tmp0.z, 0) * q;
		q = new Quaternion(tmp1.x + q.x, tmp1.y + q.y, tmp1.z + q.z, tmp1.w + q.w).normalized;

		// Part IV: Assign to the object
		transform.position = x;
		transform.rotation = q;
	}
}
