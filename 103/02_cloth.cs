using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class implicit_model : MonoBehaviour
{
	float 		dt 		= 0.0333f;
	float 		mass	= 1;
	float		damping	= 0.99f;
	float 		rho		= 0.995f;
	float 		spring_k = 8000;
	int[] 		E;
	float[] 	L;
	Vector3[] 	V;
    float[]		M;

    // Start is called before the first frame update
    void Start()
    {
		Mesh mesh = GetComponent<MeshFilter> ().mesh;

		//Resize the mesh.
		int n=21;
		Vector3[] X  	= new Vector3[n*n];
		Vector2[] UV 	= new Vector2[n*n];
		int[] triangles	= new int[(n-1)*(n-1)*6];
		for(int j=0; j<n; j++)
		for(int i=0; i<n; i++)
		{
			X[j*n+i] =new Vector3(5-10.0f*i/(n-1), 0, 5-10.0f*j/(n-1));
			UV[j*n+i]=new Vector3(i/(n-1.0f), j/(n-1.0f));
		}
		int t=0;
		for(int j=0; j<n-1; j++)
		for(int i=0; i<n-1; i++)	
		{
			triangles[t*6+0]=j*n+i;
			triangles[t*6+1]=j*n+i+1;
			triangles[t*6+2]=(j+1)*n+i+1;
			triangles[t*6+3]=j*n+i;
			triangles[t*6+4]=(j+1)*n+i+1;
			triangles[t*6+5]=(j+1)*n+i;
			t++;
		}
		mesh.vertices=X;
		mesh.triangles=triangles;
		mesh.uv = UV;
		mesh.RecalculateNormals ();


		//Construct the original E
		int[] _E = new int[triangles.Length*2];
		for (int i=0; i<triangles.Length; i+=3) 
		{
			_E[i*2+0]=triangles[i+0];
			_E[i*2+1]=triangles[i+1];
			_E[i*2+2]=triangles[i+1];
			_E[i*2+3]=triangles[i+2];
			_E[i*2+4]=triangles[i+2];
			_E[i*2+5]=triangles[i+0];
		}
		//Reorder the original edge list
		for (int i=0; i<_E.Length; i+=2)
			if(_E[i] > _E[i + 1]) 
				Swap(ref _E[i], ref _E[i+1]);
		//Sort the original edge list using quicksort
		Quick_Sort (ref _E, 0, _E.Length/2-1);

		int e_number = 0;
		for (int i=0; i<_E.Length; i+=2)
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
					e_number++;

		E = new int[e_number * 2];
		for (int i=0, e=0; i<_E.Length; i+=2)
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
			{
				E[e*2+0]=_E [i + 0];
				E[e*2+1]=_E [i + 1];
				e++;
			}

		L = new float[E.Length/2];
		for (int e=0; e<E.Length/2; e++) 
		{
			int v0 = E[e*2+0];
			int v1 = E[e*2+1];
			L[e]=(X[v0]-X[v1]).magnitude;
		}

		V = new Vector3[X.Length];
		for (int i=0; i<V.Length; i++)
			V[i] = new Vector3 (0, 0, 0);

		M = new float[X.Length];
		for (int i=0; i<M.Length; i++)
		{
			if (i == 0 || i == n - 1) M[i] = 10000000;
			else M[i] = mass;
		}
    }

    void Quick_Sort(ref int[] a, int l, int r)
	{
		int j;
		if(l<r)
		{
			j=Quick_Sort_Partition(ref a, l, r);
			Quick_Sort (ref a, l, j-1);
			Quick_Sort (ref a, j+1, r);
		}
	}

	int  Quick_Sort_Partition(ref int[] a, int l, int r)
	{
		int pivot_0, pivot_1, i, j;
		pivot_0 = a [l * 2 + 0];
		pivot_1 = a [l * 2 + 1];
		i = l;
		j = r + 1;
		while (true) 
		{
			do ++i; while( i<=r && (a[i*2]<pivot_0 || a[i*2]==pivot_0 && a[i*2+1]<=pivot_1));
			do --j; while(  a[j*2]>pivot_0 || a[j*2]==pivot_0 && a[j*2+1]> pivot_1);
			if(i>=j)	break;
			Swap(ref a[i*2], ref a[j*2]);
			Swap(ref a[i*2+1], ref a[j*2+1]);
		}
		Swap (ref a [l * 2 + 0], ref a [j * 2 + 0]);
		Swap (ref a [l * 2 + 1], ref a [j * 2 + 1]);
		return j;
	}

	void Swap(ref int a, ref int b)
	{
		int temp = a;
		a = b;
		b = temp;
	}

	void Collision_Handling()
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] X = mesh.vertices;

        //Handle colllision.
        GameObject sphere = GameObject.Find("Sphere");
		Vector3 c = sphere.transform.position;
		float r = 2.7f;
		for (int i = 0; i < V.Length; i++)
        {
            if (i == 0 || i == 20) continue;
            if ((X[i] - c).magnitude < r)
			{
				V[i] = V[i] + 1 / dt * (c + r * (X[i] - c) / (X[i] - c).magnitude - X[i]);
				X[i] = c + r * (X[i] - c) / (X[i] - c).magnitude;
			}
        }

		mesh.vertices = X;
	}

	void Get_Gradient(Vector3[] X, Vector3[] X_hat, float t, Vector3[] G)
	{
        //Momentum and Gravity.
        for (int i = 0; i < X.Length; i++) 
		{
            G[i] = (1.0f / (dt * dt)) * M[i] * (X[i] - X_hat[i]) - new Vector3(0, -9.8f, 0);
        }

        //Spring Force.
        for (int e = 0; e < E.Length / 2; e++)
        {
            int v0 = E[e * 2 + 0];
            int v1 = E[e * 2 + 1];
			Vector3 g_f = spring_k * (1 - L[e] / (X[v0] - X[v1]).magnitude) * (X[v0] - X[v1]);
			G[v0] = G[v0] + g_f;
			G[v1] = G[v1] - g_f;
        }
    }

    // Update is called once per frame
	void Update () 
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] X 			= mesh.vertices;
		Vector3[] last_X_dt		= new Vector3[X.Length];
		Vector3[] last_X_iter	= new Vector3[X.Length];
		Vector3[] X_hat			= new Vector3[X.Length];
		Vector3[] G 			= new Vector3[X.Length];

		//Initial Setup.
		for (int i = 0; i < V.Length; i++)
		{
			last_X_dt[i] = X[i];
		}

		for (int i = 0; i < V.Length; i++)
		{
			V[i] = damping * V[i];
			X_hat[i] = X[i] + dt * V[i];
			X[i] = X_hat[i];
		}

		float w = 1.0f;
		for (int k=0; k<32; k++)
		{
			if (k == 0) w = 1;
			else if (k == 1) w = 2 / (2 - rho * rho);
			else w = 4 / (4 - rho * rho * w);

			Get_Gradient(X, X_hat, dt, G);

            //Update X by gradient.
            for (int i = 0; i < X.Length; i++)
            {
				if (i ==0 || i == 20) continue;
				Vector3 tmp = X[i];
				X[i] = w * (X[i] - 1 / ((dt * dt) * mass + 4 * spring_k) * G[i]) + (1 - w) * last_X_iter[i];
				last_X_iter[i] = tmp;
            }

        }


        //Finishing.
        for (int i = 0; i < V.Length; i++) {
            V[i] = 1 / dt * (X[i] - last_X_dt[i]);
        }

        mesh.vertices = X;

		Collision_Handling ();
		mesh.RecalculateNormals ();
	}
}
