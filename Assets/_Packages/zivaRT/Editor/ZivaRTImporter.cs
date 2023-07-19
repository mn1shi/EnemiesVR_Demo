using System.Collections.Generic;
using System.IO;
using System.Linq;
using FlatBuffers;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;
using ZivaRT;

namespace Unity.ZivaRTPlayer.Editor
{
    /// <summary>
    /// Importer for ZivaRT .zrt files.
    /// To correctly import a .zrt file you should use the "Import Asset" menu item in Unity
    /// and select a .zrt and a .fbx file. The names of the files (without extension) should
    /// be identical. The importer will bind the mesh to the ZivaRT rig and create a ZivaRTPlayer
    /// object with correctly set properties.
    /// </summary>
    // The script's filename must match the class' name. If you change one, change the other as well.
    [UnityEditor.AssetImporters.ScriptedImporter(7, "zrt")]
    public class ZivaRTImporter : UnityEditor.AssetImporters.ScriptedImporter
    {
        /// <summary>
        /// The mesh that is linked to this ZRT file
        /// </summary>
        [field: SerializeField]
        [field: Tooltip("The imported mesh prefab that links to this ZRT")]
        [field: FormerlySerializedAs("<linkedFBX>k__BackingField")]
        public GameObject LinkedMesh { get; set; }

        /// <summary>
        /// Should ziva vertices be scaled by the linked mesh's imported scale setting.
        /// </summary>
        [field: SerializeField]
        [field: Tooltip("Should ziva vertices be scaled by the linked mesh's imported scale setting.")]
        public bool ScaleImportedZivaUsingLinkedMesh { get; set; } = true;


        static string[] k_FileExtensions = { "fbx", "obj" };
        /// <summary>
        /// OnImportAsset callback used for ZRT file imports.
        /// </summary>
        /// <param name="ctx">Importer context to use</param>
        public override void OnImportAsset(UnityEditor.AssetImporters.AssetImportContext ctx)
        {
            var bytes = File.ReadAllBytes(ctx.assetPath);
            ByteBuffer bb = new ByteBuffer(bytes);

            // we do not have a linked prefab currently... lets try and fix that
            // use object reference to not overrride a set, but missing linked mesh
            if (ReferenceEquals(LinkedMesh, null))
            {
                // try and find a matching mesh
                var zrtPath = ctx.assetPath;
                var zrtRoot = zrtPath.Substring(0, zrtPath.LastIndexOf('/'));

                foreach (var fileExtension in k_FileExtensions)
                {
                    var possiblePath = $"{zrtRoot}/{Path.GetFileNameWithoutExtension(zrtPath)}.{fileExtension}";
                    var loaded = AssetDatabase.LoadAssetAtPath<GameObject>(possiblePath);

                    if (loaded != null)
                    {
                        LinkedMesh = loaded;
                        break;
                    }
                }
                if (LinkedMesh is null)
                    ctx.LogImportError($"Couldn't find a corresponding fbx / obj mesh for the ZivaRT rig {zrtPath}");
            }

            var meshRescale = 1.0f;
            if (LinkedMesh != null)
            {
                var fbxPath = AssetDatabase.GetAssetPath(LinkedMesh);
                ctx.DependsOnSourceAsset(fbxPath);

                var assetImporter = GetAtPath(fbxPath);

                if (ScaleImportedZivaUsingLinkedMesh && assetImporter is ModelImporter modelImporter)
                    meshRescale = modelImporter.useFileScale ? modelImporter.fileScale : modelImporter.globalScale;
            }

            var zivaAsset = ScriptableObject.CreateInstance<ZivaRTRig>();
            zivaAsset.name = "Ziva Controller";
            var rig = Rig.GetRootAsRig(bb);
            ImportZivaCharacter(rig, zivaAsset);

            var zivaPrefab =
                LinkedMesh == null ? new GameObject(Path.GetFileName(ctx.assetPath)) : Instantiate(LinkedMesh);
            var player = zivaPrefab.AddComponent<global::ZivaRTPlayer>();
            player.Rig = zivaAsset;
            player.MeshRescale = meshRescale;

            var driverSMR = zivaPrefab.GetComponentInChildren<SkinnedMeshRenderer>();

            // if we have an FBX we can hook up a bunch of things nicely :)
            if (driverSMR != null)
            {
                player.AnimationRoot = driverSMR.rootBone;
                player.SourceMesh = driverSMR.sharedMesh;
                player.GameObjectRoot = driverSMR.transform;

                var meshRenderer = zivaPrefab.GetComponent<MeshRenderer>();
                meshRenderer.sharedMaterials = driverSMR.sharedMaterials;

                DestroyImmediate(driverSMR);
            }

            ctx.AddObjectToAsset("MainAsset", zivaPrefab);
            ctx.AddObjectToAsset("ZRT", zivaAsset);
            ctx.SetMainObject(zivaPrefab);
            player.BuildVertexOrderMapping();
            player.BuildSerializableBoundsData();
            // by default do not use custom bounds
            player.UseCustomBounds = false;
            if (player.SourceMesh != null)
            {
                // initialize to intuitive values if source mesh is present
                player.CustomBounds = new Bounds(player.SourceMesh.bounds.center, player.SourceMesh.bounds.extents);
            }
            else
            {
                player.CustomBounds = new Bounds();
            }
        }

        static void ImportZivaCharacter(Rig fbRig, ZivaRTRig rig)
        {
            rig.m_ZrtVersion = ImportVersionString(fbRig.Version.Value);

            var restPose = fbRig.RestPose.Value;
            rig.m_Character.RestShape = restPose.GetShapeArray().Clone() as float[];
            rig.m_Character.RestExtraParameters = restPose.GetExtraParametersArray().Clone() as float[];
            rig.m_Character.RestLocalTransforms = restPose.GetLocalTransformsArray().Clone() as float[];
            rig.m_Character.RestWorldTransforms = restPose.GetWorldTransformsArray().Clone() as float[];
            rig.m_Character.JointNames = ImportJointNames(restPose);

            rig.m_CorrectiveType = ConvertCorrectiveType(fbRig.Correctives.Value.CorrectiveType);
            rig.m_Patches = ImportPatches(fbRig.Correctives.Value).ToArray();
            rig.m_Skinning = ImportSkinning(fbRig.Skinning.Value);
            if (fbRig.RestPose != null)
                rig.m_ExtraParameterNames = ImportExtraParameterNames(fbRig.RestPose.Value);
        }

        static string[] ImportExtraParameterNames(ZivaRT.RestPose restPose)
        {
            string[] result = new string[restPose.ExtraParameterNamesLength];
            for (int ep = 0; ep < restPose.ExtraParameterNamesLength; ep++)
            {
                result[ep] = restPose.ExtraParameterNames(ep);
            }

            return result;
        }

        static string ImportVersionString(ZivaRT.Version version)
        {
            return string.Format("{0}.{1}.{2}", version.MajorVersion, version.MinorVersion, version.PatchVersion);
        }

        static string[] ImportJointNames(RestPose restPose)
        {
            var jointNames = new string[restPose.JointNamesLength];
            for (int j = 0; j < restPose.JointNamesLength; ++j)
            {
                jointNames[j] = restPose.JointNames(j);
            }
            return jointNames;
        }

        static IEnumerable<Patch> ImportPatches(Correctives patches)
        {
            for (int a = 0; a < patches.PatchArrayLength; a++)
            {
                yield return ImportPatch(patches.PatchArray(a).Value);
            }
        }

        static Patch ImportPatch(ZivaRT.Network zivaPatch)
        {
            var p = new Patch
            {
                Vertices = zivaPatch.GetOutputsArray().Clone() as uint[],
                PoseIndices = zivaPatch.GetPoseIndicesArray().Clone() as ushort[],
                PoseShift = zivaPatch.GetPoseShiftArray().Clone() as float[],
                PoseScale = zivaPatch.GetPoseScaleArray().Clone() as float[],
                KernelScale = zivaPatch.GetKernelScaleArray().Clone() as float[],
                KernelCenters = new MatrixX<sbyte>()
                {
                    Rows = zivaPatch.KernelCenters.Value.Rows,
                    Cols = zivaPatch.KernelCenters.Value.Cols,
                    Values = zivaPatch.KernelCenters.Value.GetXArray().Clone() as sbyte[],
                },
                ScalePerKernel = zivaPatch.GetScalePerKernelArray().Clone() as float[],
                RbfCoeffs = new MatrixX<short>()
                {
                    Rows = zivaPatch.RbfCoeffs.Value.Rows,
                    Cols = zivaPatch.RbfCoeffs.Value.Cols,
                    Values = zivaPatch.RbfCoeffs.Value.GetXArray().Clone() as short[],
                },
                ScalePerRBFCoeff = zivaPatch.GetScalePerRbfCoeffArray().Clone() as float[],
                ReducedBasis = new MatrixX<sbyte>()
                {
                    Rows = zivaPatch.ReducedBasis.Value.Rows,
                    Cols = zivaPatch.ReducedBasis.Value.Cols,
                    Values = zivaPatch.ReducedBasis.Value.GetXArray().Clone() as sbyte[],
                },
                ScalePerVertex = zivaPatch.GetScalePerVertexArray().Clone() as float[]
            };

            return p;
        }

        static CorrectiveType
        ConvertCorrectiveType(ZivaRT.CorrectiveType type)
        {
            switch (type)
            {
                case ZivaRT.CorrectiveType.TensorSkin:
                    return CorrectiveType.TensorSkin;
                case ZivaRT.CorrectiveType.EigenSkin:
                    return CorrectiveType.EigenSkin;
                case ZivaRT.CorrectiveType.FullSpace:
                    return CorrectiveType.FullSpace;
            }

            // If we get here, we didn't recognize the type. Try just casting?
            return (CorrectiveType)type;
        }

        static Skinning ImportSkinning(ZivaRT.Skinning zivaSkinning)
        {
            var s = new Skinning
            {
                RestPoseInverse = zivaSkinning.GetRestPoseInverseArray().Clone() as float[],
                SkinningWeights = ConvertSparseMatrix(zivaSkinning.SkinningWeights.Value)
            };
            return s;
        }

        static SparseMatrix ConvertSparseMatrix(ZivaRT.SparseMatrix zivaMatrix)
        {
            var s = new SparseMatrix
            {
                NumRows = zivaMatrix.NumRows,
                NumCols = zivaMatrix.NumCols,
                ColStarts = zivaMatrix.GetColStartsArray().Clone() as int[],
                RowIndices = zivaMatrix.GetRowIndicesArray().Clone() as int[],
                W = zivaMatrix.GetWArray().Clone() as float[]
            };
            return s;
        }
    }
}
