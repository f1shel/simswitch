using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Diagnostics;


public class FVM : MonoBehaviour
{
    float dt = 0.003f;
    float mass = 1;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    float damp = 0.999f;
    Vector3 gravity = new Vector3(0, -9.8f, 0);
    Vector3 floorPos = new Vector3(0, -3, 0);
    Vector3 floorNormal = new Vector3(0, 1, 0);
    float muN = 0.5f;
    float muT = 0.5f;

    int[] Tet;
    int tet_number;         //The number of tetrahedra

    Vector3[] Force;
    Vector3[] V;
    Vector3[] X;
    int number;             //The number of vertices

    Matrix4x4[] inv_Dm;

    //For Laplacian smoothing.
    Vector3[] V_sum;
    int[] V_num;

    SVD svd = new SVD();

    // Start is called before the first frame update
    void Start()
    {
        // FILO IO: Read the house model from files.
        // The model is from Jonathan Schewchuk's Stellar lib.
        {
            string fileContent = File.ReadAllText("Assets/house2.ele");
            string[] Strings = fileContent.Split(new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            tet_number = int.Parse(Strings[0]);
            Tet = new int[tet_number * 4];

            for (int tet = 0; tet < tet_number; tet++)
            {
                Tet[tet * 4 + 0] = int.Parse(Strings[tet * 5 + 4]) - 1;
                Tet[tet * 4 + 1] = int.Parse(Strings[tet * 5 + 5]) - 1;
                Tet[tet * 4 + 2] = int.Parse(Strings[tet * 5 + 6]) - 1;
                Tet[tet * 4 + 3] = int.Parse(Strings[tet * 5 + 7]) - 1;
            }
        }
        {
            string fileContent = File.ReadAllText("Assets/house2.node");
            string[] Strings = fileContent.Split(new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            number = int.Parse(Strings[0]);
            X = new Vector3[number];
            for (int i = 0; i < number; i++)
            {
                X[i].x = float.Parse(Strings[i * 5 + 5]) * 0.4f;
                X[i].y = float.Parse(Strings[i * 5 + 6]) * 0.4f;
                X[i].z = float.Parse(Strings[i * 5 + 7]) * 0.4f;
            }
            //Centralize the model.
            Vector3 center = Vector3.zero;
            for (int i = 0; i < number; i++) center += X[i];
            center = center / number;
            for (int i = 0; i < number; i++)
            {
                X[i] -= center;
                float temp = X[i].y;
                X[i].y = X[i].z;
                X[i].z = temp;
            }
        }
        // tet_number = 1; // number of tetrahedrons
        // Tet = new int[tet_number * 4];
        // Tet[0] = 0;
        // Tet[1] = 1;
        // Tet[2] = 2;
        // Tet[3] = 3;

        // number = 4; // number of vertices of tetrahedrons
        // X = new Vector3[number];
        // V = new Vector3[number];
        // Force = new Vector3[number];
        // X[0] = new Vector3(0, 0, 0);
        // X[1] = new Vector3(1, 0, 0);
        // X[2] = new Vector3(0, 1, 0);
        // X[3] = new Vector3(0, 0, 1);


        //Create triangle mesh.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }

        int[] triangles = new int[tet_number * 12];
        for (int t = 0; t < tet_number * 4; t++)
        {
            triangles[t * 3 + 0] = t * 3 + 0;
            triangles[t * 3 + 1] = t * 3 + 1;
            triangles[t * 3 + 2] = t * 3 + 2;
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        // for (int t = 0; t < tet_number * 4; t++)
        // {
        //     print(triangles[t * 3 + 0]);
        //     print(triangles[t * 3 + 1]);
        //     print(triangles[t * 3 + 2]);
        // }
        mesh.RecalculateNormals();


        V = new Vector3[number];
        Force = new Vector3[number];
        V_sum = new Vector3[number];
        V_num = new int[number];

        //TODO: Need to allocate and assign inv_Dm
        // print(vertices.Length);
        inv_Dm = new Matrix4x4[tet_number];
        for (int t = 0; t < tet_number; t++)
        {
            Matrix4x4 edge_matrix = Build_Edge_Matrix(t);
            inv_Dm[t] = edge_matrix.inverse;
        }
    }

    Matrix4x4 Build_Edge_Matrix(int tet)
    {
        Matrix4x4 ret = Matrix4x4.zero;
        //TODO: Need to build edge matrix here.
        Vector3 X0 = X[Tet[tet * 4 + 0]];
        Vector3 X1 = X[Tet[tet * 4 + 1]];
        Vector3 X2 = X[Tet[tet * 4 + 2]];
        Vector3 X3 = X[Tet[tet * 4 + 3]];
        return Build_Edge_Matrix__(X0, X1, X2, X3);
    }
    Matrix4x4 Build_Edge_Matrix__(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3)
    {
        Matrix4x4 ret = Matrix4x4.zero;
        //TODO: Need to build edge matrix here.
        var v10 = v1 - v0;
        var v20 = v2 - v0;
        var v30 = v3 - v0;
        ret.SetColumn(0, (Vector4)v10);
        ret.SetColumn(1, (Vector4)v20);
        ret.SetColumn(2, (Vector4)v30);
        ret.SetColumn(3, new Vector4(0, 0, 0, 1));
        return ret;
    }

    void _Update()
    {
        // Jump up.
        if (Input.GetKeyDown(KeyCode.Space))
        {
            for (int i = 0; i < number; i++)
                V[i].y += 0.2f;

        }

        for (int i = 0; i < number; i++)
        {
            //TODO: Add gravity to Force.
            Force[i] = mass * gravity;
        }

        Vector3[] vertices = GetComponent<MeshFilter>().mesh.vertices;
        for (int tet = 0; tet < tet_number; tet++)
        {
            //TODO: Deformation Gradient
            Matrix4x4 F = Build_Edge_Matrix(tet) * inv_Dm[tet];

            //TODO: Green Strain
            Matrix4x4 G = F.transpose * F;
            Matrix4x4 I = Matrix4x4.identity;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    G[i, j] = 0.5f * (G[i, j] - I[i, j]);
            //print("G");
            //print(G);

            //TODO: Second PK Stress
            float trG = G[0, 0] + G[1, 1] + G[2, 2];
            Matrix4x4 G1 = G;
            Matrix4x4 G2 = I;
            Matrix4x4 S = Matrix4x4.identity;
            for (int i = 0; i < 3; i++)
            {
                G2[i, i] *= stiffness_0 * trG;
                for (int j = 0; j < 3; j++)
                {
                    G1[i, j] *= 2 * stiffness_1;
                    S[i, j] = G1[i, j] + G2[i, j];
                }
            }
            //print("S");
            //print(S);

            //TODO: Elastic Force
            Matrix4x4 P = F * S;
            Matrix4x4 force_matrix = P * inv_Dm[tet].transpose;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    force_matrix[i, j] *= -1 / (6 * inv_Dm[tet].determinant);
            Vector4 f1 = force_matrix.GetColumn(0);
            Vector4 f2 = force_matrix.GetColumn(1);
            Vector4 f3 = force_matrix.GetColumn(2);
            Vector4 f0 = -(f1 + f2 + f3);
            Force[Tet[tet * 4 + 0]] += (Vector3)f0;
            Force[Tet[tet * 4 + 1]] += (Vector3)f1;
            Force[Tet[tet * 4 + 2]] += (Vector3)f2;
            Force[Tet[tet * 4 + 3]] += (Vector3)f3;
        }

        for (int i = 0; i < number; i++)
        {
            //TODO: Update X and V here.
            V[i] *= damp;
            V[i] += Force[i] / mass * dt;

        }

        Laplacian_Smoothing();

        for (int i = 0; i < number; i++)
        {
            X[i] += V[i] * dt;

            //TODO: (Particle) collision with floor.
            float signedDis = Vector3.Dot(X[i] - floorPos, floorNormal);
            if (signedDis < 0 && Vector3.Dot(V[i], floorNormal) < 0)
            {
                X[i] -= signedDis * floorNormal;
                Vector3 vN = Vector3.Dot(V[i], floorNormal) * floorNormal;
                Vector3 vT = V[i] - vN;
                float a = Math.Max(1 - muT * (1 + muN) * vN.magnitude / vT.magnitude, 0);
                V[i] = -muN * vN + a * vT;
            }
        }
    }

    void Laplacian_Smoothing()
    {
        for (int i = 0; i < number; i++)
        {
            V_sum[i] = new Vector3(0, 0, 0);
            V_num[i] = 0;
        }

        for (int tet = 0; tet < tet_number; tet++)
        {
            Vector3 tmp = V[Tet[tet * 4 + 0]] + V[Tet[tet * 4 + 1]] + V[Tet[tet * 4 + 2]] + V[Tet[tet * 4 + 3]];
            for (int i = 0; i < 4; i++)
            {
                V_num[Tet[tet * 4 + i]] += 3;
                V_sum[Tet[tet * 4 + i]] += tmp - V[Tet[tet * 4 + i]];
            }
        }
        float blendRatio = 0.5f;
        for (int i = 0; i < number; i++)
        {
            V[i] = V[i] * blendRatio + V_sum[i] / V_num[i] * (1 - blendRatio);
        }
    }

    // Update is called once per frame
    void Update()
    {
        for (int l = 0; l < 10; l++)
            _Update();

        // Dump the vertex array for rendering.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = vertices;
        mesh.RecalculateNormals();
    }
}
