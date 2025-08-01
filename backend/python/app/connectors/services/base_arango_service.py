"""ArangoDB service for interacting with the database"""

# pylint: disable=E1101, W0718
import asyncio
import uuid
from typing import Dict, List, Tuple

from arango import ArangoClient, Optional

from app.config.configuration_service import (
    ConfigurationService,
    DefaultEndpoints,
    config_node_constants,
)
from app.config.utils.named_constants.arangodb_constants import (
    CollectionNames,
    Connectors,
    DepartmentNames,
    OriginTypes,
    RecordTypes,
)
from app.connectors.services.kafka_service import KafkaService
from app.schema.arango.documents import (
    app_schema,
    department_schema,
    file_record_schema,
    kb_schema,
    mail_record_schema,
    orgs_schema,
    record_group_schema,
    record_schema,
    user_schema,
)
from app.schema.arango.edges import (
    basic_edge_schema,
    belongs_to_schema,
    is_of_type_schema,
    permissions_schema,
    record_relations_schema,
    user_app_relation_schema,
    user_drive_relation_schema,
)
from app.utils.time_conversion import get_epoch_timestamp_in_ms

# Collection definitions with their schemas
NODE_COLLECTIONS = [
    (CollectionNames.RECORDS.value, record_schema),
    (CollectionNames.DRIVES.value, None),
    (CollectionNames.FILES.value, file_record_schema),
    (CollectionNames.LINKS.value, None),
    (CollectionNames.MAILS.value, mail_record_schema),
    (CollectionNames.PEOPLE.value, None),
    (CollectionNames.USERS.value, user_schema),
    (CollectionNames.GROUPS.value, None),
    (CollectionNames.ORGS.value, orgs_schema),
    (CollectionNames.ANYONE.value, None),
    (CollectionNames.CHANNEL_HISTORY.value, None),
    (CollectionNames.PAGE_TOKENS.value, None),
    (CollectionNames.APPS.value, app_schema),
    (CollectionNames.DEPARTMENTS.value, department_schema),
    (CollectionNames.CATEGORIES.value, None),
    (CollectionNames.LANGUAGES.value, None),
    (CollectionNames.TOPICS.value, None),
    (CollectionNames.SUBCATEGORIES1.value, None),
    (CollectionNames.SUBCATEGORIES2.value, None),
    (CollectionNames.SUBCATEGORIES3.value, None),
    (CollectionNames.BLOCKS.value, None),
    (CollectionNames.KNOWLEDGE_BASE.value, kb_schema),
    (CollectionNames.RECORD_GROUPS.value, record_group_schema)
]

EDGE_COLLECTIONS = [
    (CollectionNames.IS_OF_TYPE.value, is_of_type_schema),
    (CollectionNames.RECORD_RELATIONS.value, record_relations_schema),
    (CollectionNames.USER_DRIVE_RELATION.value, user_drive_relation_schema),
    (CollectionNames.BELONGS_TO_DEPARTMENT.value, basic_edge_schema),
    (CollectionNames.ORG_DEPARTMENT_RELATION.value, basic_edge_schema),
    (CollectionNames.BELONGS_TO.value, belongs_to_schema),
    (CollectionNames.PERMISSIONS.value, permissions_schema),
    (CollectionNames.ORG_APP_RELATION.value, basic_edge_schema),
    (CollectionNames.USER_APP_RELATION.value, user_app_relation_schema),
    (CollectionNames.BELONGS_TO_CATEGORY.value, basic_edge_schema),
    (CollectionNames.BELONGS_TO_LANGUAGE.value, basic_edge_schema),
    (CollectionNames.BELONGS_TO_TOPIC.value, basic_edge_schema),
    (CollectionNames.INTER_CATEGORY_RELATIONS.value, basic_edge_schema),
    (CollectionNames.BELONGS_TO_KNOWLEDGE_BASE.value, belongs_to_schema),
    (CollectionNames.PERMISSIONS_TO_KNOWLEDGE_BASE.value, permissions_schema),
    (CollectionNames.BELONGS_TO_KB.value, belongs_to_schema),
    (CollectionNames.PERMISSIONS_TO_KB.value, permissions_schema),
]

class BaseArangoService:
    """Base ArangoDB service class for interacting with the database"""

    def __init__(
        self, logger, arango_client: ArangoClient, config: ConfigurationService, kafka_service: KafkaService,
    ) -> None:
        self.logger = logger
        self.config = config
        self.client = arango_client
        self.kafka_service = kafka_service
        self.db = None

        self.connector_delete_permissions = {
            Connectors.GOOGLE_DRIVE.value: {
                "allowed_roles": ["OWNER", "WRITER", "FILEORGANIZER"],
                "edge_collections": [
                    CollectionNames.IS_OF_TYPE.value,
                    CollectionNames.RECORD_RELATIONS.value,
                    CollectionNames.PERMISSIONS.value,
                    CollectionNames.USER_DRIVE_RELATION.value,
                    CollectionNames.BELONGS_TO.value,
                    CollectionNames.ANYONE.value
                ],
                "document_collections": [
                    CollectionNames.RECORDS.value,
                    CollectionNames.FILES.value,
                ]
            },
            Connectors.GOOGLE_MAIL.value: {
                "allowed_roles": ["OWNER", "WRITER"],
                "edge_collections": [
                    CollectionNames.IS_OF_TYPE.value,
                    CollectionNames.RECORD_RELATIONS.value,
                    CollectionNames.PERMISSIONS.value,
                    CollectionNames.BELONGS_TO.value,
                ],
                "document_collections": [
                    CollectionNames.RECORDS.value,
                    CollectionNames.MAILS.value,
                    CollectionNames.FILES.value,  # For attachments
                ]
            },
            Connectors.KNOWLEDGE_BASE.value: {
                "allowed_roles": ["OWNER", "WRITER", "FILEORGANIZER"],
                "edge_collections": [
                    CollectionNames.IS_OF_TYPE.value,
                    CollectionNames.RECORD_RELATIONS.value,
                    CollectionNames.BELONGS_TO_KB.value,
                    CollectionNames.PERMISSIONS_TO_KB.value,
                ],
                "document_collections": [
                    CollectionNames.RECORDS.value,
                    CollectionNames.FILES.value,
                    CollectionNames.RECORD_GROUPS.value,
                ]
            }
        }

        # Initialize collections dictionary
        self._collections = {
            collection_name: None
            for collection_name, _ in NODE_COLLECTIONS + EDGE_COLLECTIONS
        }

    async def connect(self) -> bool:
        """Connect to ArangoDB and initialize collections"""
        try:
            self.logger.info("🚀 Connecting to ArangoDB...")
            arangodb_config = await self.config.get_config(
                config_node_constants.ARANGODB.value
            )
            arango_url = arangodb_config["url"]
            arango_user = arangodb_config["username"]
            arango_password = arangodb_config["password"]
            arango_db = arangodb_config["db"]

            if not isinstance(arango_url, str):
                raise ValueError("ArangoDB URL must be a string")
            if not self.client:
                return False

            # Connect to system db to ensure our db exists
            self.logger.debug("Connecting to system db")
            sys_db = self.client.db(
                "_system", username=arango_user, password=arango_password, verify=True
            )
            self.logger.debug("System DB: %s", sys_db)

            # Check if database exists, but don't try to create if it does
            self.logger.debug("Checking if our database exists")
            if not sys_db.has_database(arango_db):
                try:
                    self.logger.info(
                        "🚀 Database %s does not exist. Creating...", arango_db
                    )
                    sys_db.create_database(arango_db)
                    self.logger.info("✅ Database created successfully")
                except Exception as e:
                    # If database creation fails but database exists, we can continue
                    if "duplicate database name" not in str(e):
                        raise
                    self.logger.warning(
                        "Database already exists, continuing with connection"
                    )

            # Connect to our database
            self.logger.debug("Connecting to our database")
            self.db = self.client.db(
                arango_db, username=arango_user, password=arango_password, verify=True
            )
            self.logger.debug("Our DB: %s", self.db)

            # Initialize collections with schema update handling
            try:
                # Initialize all collections (both nodes and edges)
                for collection_name, schema in NODE_COLLECTIONS + EDGE_COLLECTIONS:
                    is_edge = (collection_name, schema) in EDGE_COLLECTIONS

                    collection = self._collections[collection_name] = (
                        self.db.collection(collection_name)
                        if self.db.has_collection(collection_name)
                        else self.db.create_collection(
                            collection_name,
                            edge=is_edge,
                            schema=schema
                        )
                    )

                    # Update schema if collection exists and has a schema
                    if self.db.has_collection(collection_name) and schema:
                        try:
                            self.logger.info(f"Updating schema for collection {collection_name}")
                            collection.configure(schema=schema)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to update schema for {collection_name}: {str(e)}"
                            )

                # Create the permissions graph if it doesn't exist
                if not self.db.has_graph(CollectionNames.FILE_ACCESS_GRAPH.value):
                    self.logger.info("🚀 Creating file access graph...")
                    graph = self.db.create_graph(CollectionNames.FILE_ACCESS_GRAPH.value)

                    # Define edge definitions
                    edge_definitions = [
                        {
                            "edge_collection": CollectionNames.PERMISSIONS.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [
                                CollectionNames.USERS.value,
                                CollectionNames.GROUPS.value,
                                CollectionNames.ORGS.value,
                            ],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO.value,
                            "from_vertex_collections": [CollectionNames.USERS.value],
                            "to_vertex_collections": [
                                CollectionNames.GROUPS.value,
                                CollectionNames.ORGS.value,
                            ],
                        },
                        {
                            "edge_collection": CollectionNames.ORG_DEPARTMENT_RELATION.value,
                            "from_vertex_collections": [CollectionNames.ORGS.value],
                            "to_vertex_collections": [CollectionNames.DEPARTMENTS.value],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_DEPARTMENT.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [CollectionNames.DEPARTMENTS.value],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_CATEGORY.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [
                                CollectionNames.CATEGORIES.value,
                                CollectionNames.SUBCATEGORIES1.value,
                                CollectionNames.SUBCATEGORIES2.value,
                                CollectionNames.SUBCATEGORIES3.value,
                            ],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_TOPIC.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [CollectionNames.TOPICS.value],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_LANGUAGE.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [CollectionNames.LANGUAGES.value],
                        },
                        {
                            "edge_collection": CollectionNames.INTER_CATEGORY_RELATIONS.value,
                            "from_vertex_collections": [CollectionNames.CATEGORIES.value, CollectionNames.SUBCATEGORIES1.value, CollectionNames.SUBCATEGORIES2.value, CollectionNames.SUBCATEGORIES3.value],
                            "to_vertex_collections": [CollectionNames.CATEGORIES.value, CollectionNames.SUBCATEGORIES1.value, CollectionNames.SUBCATEGORIES2.value, CollectionNames.SUBCATEGORIES3.value],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_KNOWLEDGE_BASE.value,   # record belongs to KB
                            "from_vertex_collections": [CollectionNames.RECORDS.value,CollectionNames.FILES.value],
                            "to_vertex_collections": [CollectionNames.KNOWLEDGE_BASE.value],
                        },
                        {
                            "edge_collection": CollectionNames.PERMISSIONS_TO_KNOWLEDGE_BASE.value,   # user KB permission
                            "from_vertex_collections": [CollectionNames.USERS.value],
                            "to_vertex_collections": [CollectionNames.KNOWLEDGE_BASE.value],
                        },
                        {
                            "edge_collection": CollectionNames.BELONGS_TO_KB.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [CollectionNames.RECORD_GROUPS.value],
                        },
                        {
                            "edge_collection": CollectionNames.PERMISSIONS_TO_KB.value,
                            "from_vertex_collections": [CollectionNames.USERS.value],
                            "to_vertex_collections": [CollectionNames.RECORD_GROUPS.value],
                        },
                        {
                            "edge_collection": CollectionNames.IS_OF_TYPE.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value],
                            "to_vertex_collections": [CollectionNames.FILES.value],
                        },
                        {
                            "edge_collection": CollectionNames.RECORD_RELATIONS.value,
                            "from_vertex_collections": [CollectionNames.RECORDS.value, CollectionNames.FILES.value,CollectionNames.KNOWLEDGE_BASE.value],
                            "to_vertex_collections": [CollectionNames.RECORDS.value, CollectionNames.FILES.value],
                        },
                        {
                            "edge_collection": CollectionNames.USER_DRIVE_RELATION.value,
                            "from_vertex_collections": [CollectionNames.USERS.value],
                            "to_vertex_collections": [CollectionNames.DRIVES.value],
                        },
                        {
                            "edge_collection": CollectionNames.USER_APP_RELATION.value,
                            "from_vertex_collections": [CollectionNames.USERS.value],
                            "to_vertex_collections": [CollectionNames.APPS.value],
                        },
                        {
                            "edge_collection": CollectionNames.ORG_APP_RELATION.value,
                            "from_vertex_collections": [CollectionNames.ORGS.value],
                            "to_vertex_collections": [CollectionNames.APPS.value],
                        },
                    ]

                    # Create all edge definitions
                    for edge_def in edge_definitions:
                        graph.create_edge_definition(**edge_def)

                    self.logger.info("✅ File access graph created successfully")

                self.logger.info("✅ Collections initialized successfully")

                # Initialize departments
                try:
                    await self._initialize_departments()
                except Exception as e:
                    self.logger.error("❌ Error initializing departments: %s", str(e))
                    raise

                return True

            except Exception as e:
                self.logger.error("❌ Error initializing collections: %s", str(e))
                raise

        except Exception as e:
            self.logger.error("❌ Failed to connect to ArangoDB: %s", str(e))
            self.client = None
            self.db = None
            # Reset collections
            for collection in self._collections:
                self._collections[collection] = None
            return False

    async def _initialize_departments(self) -> None:
        """Initialize departments collection with predefined department types"""
        departments = [
            {
                "_key": str(uuid.uuid4()),
                "departmentName": dept.value,
                "orgId": None,
            }
            for dept in DepartmentNames
        ]

        # Bulk insert departments if not already present
        existing_department_names = set(
            doc["departmentName"]
            for doc in self._collections[CollectionNames.DEPARTMENTS.value].all()
        )

        new_departments = [
            dept
            for dept in departments
            if dept["departmentName"] not in existing_department_names
        ]

        if new_departments:
            self.logger.info(f"🚀 Inserting {len(new_departments)} departments")
            self._collections[CollectionNames.DEPARTMENTS.value].insert_many(
                new_departments
            )
            self.logger.info("✅ Departments initialized successfully")

    async def disconnect(self) -> bool | None:
        """Disconnect from ArangoDB"""
        try:
            self.logger.info("🚀 Disconnecting from ArangoDB")
            if self.client:
                self.client.close()
            self.logger.info("✅ Disconnected from ArangoDB successfully")
        except Exception as e:
            self.logger.error("❌ Failed to disconnect from ArangoDB: %s", str(e))
            return False

    async def get_org_apps(self, org_id: str) -> list:
        """Get all apps associated with an organization"""
        try:
            query = f"""
            FOR app IN OUTBOUND
                '{CollectionNames.ORGS.value}/{org_id}'
                {CollectionNames.ORG_APP_RELATION.value}
            FILTER app.isActive == true
            RETURN app
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Failed to get org apps: {str(e)}")
            raise

    async def get_user_apps(self, user_id: str) -> list:
        """Get all apps associated with a user"""
        try:
            query = f"""
            FOR app IN OUTBOUND
                '{CollectionNames.USERS.value}/{user_id}'
                {CollectionNames.USER_APP_RELATION.value}
            RETURN app
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Failed to get user apps: {str(e)}")
            raise

    async def get_all_orgs(self, active: bool = True) -> list:
        """Get all organizations, optionally filtering by active status."""
        try:
            query = f"""
            FOR org IN {CollectionNames.ORGS.value}
            FILTER @active == false || org.isActive == true
            RETURN org
            """

            bind_vars = {"active": active}

            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Failed to get organizations: {str(e)}")
            raise

    async def get_document(self, document_key: str, collection: str) -> Optional[Dict]:
            """Get a document by its key"""
            try:
                query = """
                FOR doc IN @@collection
                    FILTER doc._key == @document_key
                    RETURN doc
                """
                cursor = self.db.aql.execute(
                    query,
                    bind_vars={"document_key": document_key, "@collection": collection},
                )
                result = list(cursor)
                return result[0] if result else None
            except Exception as e:
                self.logger.error("❌ Error getting document: %s", str(e))
                return None

    async def get_user_by_user_id(self, user_id: str) -> Optional[Dict]:
        """Get user by user ID"""
        try:
            query = f"""
                FOR user IN {CollectionNames.USERS.value}
                    FILTER user.userId == @user_id
                    RETURN user
            """
            cursor = self.db.aql.execute(query, bind_vars={"user_id": user_id})
            result = next(cursor, None)
            return result
        except Exception as e:
            self.logger.error(f"Error getting user by user ID: {str(e)}")
            return None

    async def get_connector_stats(
        self,
        org_id: str,
        user_id: str,
    ) -> Dict:
        """
        Get comprehensive connector statistics for an organization
        Complete replacement for Node.js getConnectorStats - works with all connectors
        """
        try:
            self.logger.info(f"🔍 Getting connector stats for organization: {org_id}")

            # Get user for permission filtering
            user = await self.get_user_by_user_id(user_id=user_id)
            if not user:
                return {
                    "success": False,
                    "message": "User not found",
                    "data": None
                }

            user_key = user.get('_key')
            db = self.db

            # Single comprehensive query that handles all statistics
            query = """
            LET org_id = @org_id
            LET user_from = @user_from
            // Get all records that match the base filter (excluding DRIVE type)
            LET all_org_records = (
                FOR doc IN @@records
                    FILTER doc.orgId == org_id AND doc.recordType != "DRIVE"
                    RETURN doc
            )
            // Get accessible KB records (UPLOAD origin)
            LET accessible_kb_records = (
                FOR kbEdge IN @@permissions_to_kb
                    FILTER kbEdge._from == user_from
                    FILTER kbEdge.type == "USER"
                    FILTER kbEdge.role IN ["OWNER", "READER", "FILEORGANIZER", "WRITER", "COMMENTER", "ORGANIZER"]
                    LET kb = DOCUMENT(kbEdge._to)
                    FILTER kb != null AND kb.orgId == org_id
                    FOR belongsEdge IN @@belongs_to_kb
                        FILTER belongsEdge._to == kb._id
                        LET record = DOCUMENT(belongsEdge._from)
                        FILTER record != null
                        FILTER record.isDeleted != true
                        FILTER record.orgId == org_id
                        FILTER record.origin == "UPLOAD"
                        FILTER record.recordType != "DRIVE"
                        RETURN {
                            record: record,
                            kb_id: kb._key,
                            kb_name: kb.groupName
                        }
            )
            // Get accessible connector records (CONNECTOR origin)
            LET accessible_connector_records = (
                FOR permissionEdge IN @@permissions
                    FILTER permissionEdge._to == user_from
                    LET record = DOCUMENT(permissionEdge._from)
                    FILTER record != null
                    FILTER record.isDeleted != true
                    FILTER record.orgId == org_id
                    FILTER record.origin == "CONNECTOR"
                    FILTER record.recordType != "DRIVE"
                    RETURN record
            )
            // Combine all accessible records
            LET kb_records_only = accessible_kb_records[*].record
            LET all_accessible_records = APPEND(kb_records_only, accessible_connector_records)
            // Overall statistics (across all accessible records)
            LET total_stats = {
                total: LENGTH(all_accessible_records),
                indexing_status: {
                    NOT_STARTED: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                    IN_PROGRESS: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                    COMPLETED: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                    FAILED: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                    FILE_TYPE_NOT_SUPPORTED: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                    AUTO_INDEX_OFF: LENGTH(all_accessible_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                }
            }
            // Overall connector statistics (CONNECTOR origin only)
            LET overall_connector_stats = {
                total: LENGTH(accessible_connector_records),
                indexing_status: {
                    NOT_STARTED: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                    IN_PROGRESS: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                    COMPLETED: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                    FAILED: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                    FILE_TYPE_NOT_SUPPORTED: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                    AUTO_INDEX_OFF: LENGTH(accessible_connector_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                }
            }
            // Upload statistics (UPLOAD origin only - KB records)
            LET upload_stats = {
                total: LENGTH(kb_records_only),
                indexing_status: {
                    NOT_STARTED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                    IN_PROGRESS: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                    COMPLETED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                    FAILED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "FAILED"]),
                    FILE_TYPE_NOT_SUPPORTED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                    AUTO_INDEX_OFF: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                }
            }
            // Enhanced connector stats with record type breakdowns
            LET connector_data = (
                // Regular connectors (CONNECTOR origin)
                FOR connector_name IN UNIQUE(accessible_connector_records[*].connectorName)
                    FILTER connector_name != null
                    LET connector_records = accessible_connector_records[* FILTER CURRENT.connectorName == connector_name]
                    // Record type breakdown for this connector
                    LET record_types = (
                        FOR record_type IN UNIQUE(connector_records[*].recordType)
                            LET type_records = connector_records[* FILTER CURRENT.recordType == record_type]
                            RETURN {
                                record_type: record_type,
                                total: LENGTH(type_records),
                                indexing_status: {
                                    NOT_STARTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                                    IN_PROGRESS: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                                    COMPLETED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                                    FAILED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                                    FILE_TYPE_NOT_SUPPORTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                                    AUTO_INDEX_OFF: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                                }
                            }
                    )
                    RETURN {
                        connector: connector_name,
                        total: LENGTH(connector_records),
                        indexing_status: {
                            NOT_STARTED: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                            IN_PROGRESS: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                            COMPLETED: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                            FAILED: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                            FILE_TYPE_NOT_SUPPORTED: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                            AUTO_INDEX_OFF: LENGTH(connector_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                        },
                        by_record_type: record_types
                    }
            )
            // Add Knowledge Base as a connector if there are KB records
            LET kb_connector_data = LENGTH(kb_records_only) > 0 ? [{
                connector: "KNOWLEDGE_BASE",
                total: LENGTH(kb_records_only),
                indexing_status: {
                    NOT_STARTED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                    IN_PROGRESS: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                    COMPLETED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                    FAILED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "FAILED"]),
                    FILE_TYPE_NOT_SUPPORTED: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                    AUTO_INDEX_OFF: LENGTH(kb_records_only[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                },
                by_record_type: (
                    FOR record_type IN UNIQUE(kb_records_only[*].recordType)
                        LET type_records = kb_records_only[* FILTER CURRENT.recordType == record_type]
                        RETURN {
                            record_type: record_type,
                            total: LENGTH(type_records),
                            indexing_status: {
                                NOT_STARTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                                IN_PROGRESS: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                                COMPLETED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                                FAILED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                                FILE_TYPE_NOT_SUPPORTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                                AUTO_INDEX_OFF: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                            }
                        }
                ),
                knowledge_bases: (
                    FOR kb_item IN accessible_kb_records
                        COLLECT kb_id = kb_item.kb_id, kb_name = kb_item.kb_name INTO kb_group = kb_item
                        LET kb_records = kb_group[*].record
                        // Record types within this KB
                        LET kb_record_types = (
                            FOR kb_record_type IN UNIQUE(kb_records[*].recordType)
                                LET kb_type_records = kb_records[* FILTER CURRENT.recordType == kb_record_type]
                                RETURN {
                                    record_type: kb_record_type,
                                    total: LENGTH(kb_type_records),
                                    indexing_status: {
                                        NOT_STARTED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                                        IN_PROGRESS: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                                        COMPLETED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                                        FAILED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                                        FILE_TYPE_NOT_SUPPORTED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                                        AUTO_INDEX_OFF: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                                    }
                                }
                        )
                        RETURN {
                            kb_id: kb_id,
                            kb_name: kb_name,
                            total: LENGTH(kb_records),
                            indexing_status: {
                                NOT_STARTED: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                                IN_PROGRESS: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                                COMPLETED: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                                FAILED: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                                FILE_TYPE_NOT_SUPPORTED: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                                AUTO_INDEX_OFF: LENGTH(kb_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                            },
                            by_record_type: kb_record_types
                        }
                )
            }] : []
            // Combine all connector data
            LET all_connector_data = APPEND(connector_data, kb_connector_data)
            // Stats by record type across all connectors
            LET record_type_stats = (
                FOR record_type IN UNIQUE(all_accessible_records[*].recordType)
                    LET type_records = all_accessible_records[* FILTER CURRENT.recordType == record_type]
                    // Connector breakdown for this record type
                    LET connectors_for_type = (
                        // Connector records
                        FOR connector_name IN UNIQUE(accessible_connector_records[* FILTER CURRENT.recordType == record_type][*].connectorName)
                            FILTER connector_name != null
                            LET connector_type_records = accessible_connector_records[* FILTER CURRENT.recordType == record_type AND CURRENT.connectorName == connector_name]
                            RETURN {
                                connector: connector_name,
                                total: LENGTH(connector_type_records),
                                indexing_status: {
                                    NOT_STARTED: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                                    IN_PROGRESS: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                                    COMPLETED: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                                    FAILED: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                                    FILE_TYPE_NOT_SUPPORTED: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                                    AUTO_INDEX_OFF: LENGTH(connector_type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                                }
                            }
                    )
                    // Add Knowledge Base connector if there are KB records of this type
                    LET kb_type_records = kb_records_only[* FILTER CURRENT.recordType == record_type]
                    LET kb_connector_for_type = LENGTH(kb_type_records) > 0 ? [{
                        connector: "KNOWLEDGE_BASE",
                        total: LENGTH(kb_type_records),
                        indexing_status: {
                            NOT_STARTED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                            IN_PROGRESS: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                            COMPLETED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                            FAILED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                            FILE_TYPE_NOT_SUPPORTED: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                            AUTO_INDEX_OFF: LENGTH(kb_type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                        }
                    }] : []
                    LET all_connectors_for_type = APPEND(connectors_for_type, kb_connector_for_type)
                    RETURN {
                        record_type: record_type,
                        total: LENGTH(type_records),
                        indexing_status: {
                            NOT_STARTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "NOT_STARTED"]),
                            IN_PROGRESS: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "IN_PROGRESS"]),
                            COMPLETED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "COMPLETED"]),
                            FAILED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FAILED"]),
                            FILE_TYPE_NOT_SUPPORTED: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "FILE_TYPE_NOT_SUPPORTED"]),
                            AUTO_INDEX_OFF: LENGTH(type_records[* FILTER CURRENT.indexingStatus == "AUTO_INDEX_OFF"])
                        },
                        by_connector: all_connectors_for_type
                    }
            )
            // Return comprehensive stats (matching Node.js structure exactly)
            RETURN {
                org_id: org_id,
                total: total_stats,
                overall_connector: overall_connector_stats,
                upload: upload_stats,
                by_connector: all_connector_data,
                by_record_type: record_type_stats
            }
            """

            # Execute the comprehensive query
            cursor = db.aql.execute(query, bind_vars={
                "org_id": org_id,
                "user_from": f"users/{user_key}",
                "@records": CollectionNames.RECORDS.value,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
            })

            result = next(cursor, None)

            # Return the result in the same format as Node.js
            if result:
                self.logger.info(f"✅ Retrieved enhanced connector stats for organization: {org_id}")
                return {
                    "success": True,
                    "data": result
                }
            else:
                self.logger.warning(f"⚠️ No data found for organization: {org_id}")
                return {
                    "success": False,
                    "message": "No data found",
                    "data": None
                }

        except Exception as e:
            self.logger.error(f"❌ Error getting connector stats: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }

    async def check_record_access_with_details(
        self, user_id: str, org_id: str, record_id: str
    ) -> Optional[Dict]:
        """
        Check record access and return record details if accessible
        Args:
            user_id (str): The userId field value in users collection
            org_id (str): The organization ID
            record_id (str): The record ID to check access for
        Returns:
            dict: Record details with permissions if accessible, None if not
        """
        try:
            # First check access and get permission paths
            access_query = f"""
            LET userDoc = FIRST(
                FOR user IN @@users
                FILTER user.userId == @userId
                RETURN user
            )
            LET recordDoc = DOCUMENT(CONCAT(@records, '/', @recordId))
            LET kb = FIRST(
                FOR k IN 1..1 OUTBOUND recordDoc._id @@belongs_to_kb
                RETURN k
            )
            LET directAccess = (
                FOR records, edge IN 1..1 ANY userDoc._id {CollectionNames.PERMISSIONS.value}
                FILTER records._key == @recordId
                RETURN {{
                    type: 'DIRECT',
                    source: userDoc,
                    role: edge.role
                }}
            )
            LET groupAccess = (
                FOR group, belongsEdge IN 1..1 ANY userDoc._id {CollectionNames.BELONGS_TO.value}
                FILTER belongsEdge.entityType == 'GROUP'
                FOR records, permEdge IN 1..1 ANY group._id {CollectionNames.PERMISSIONS.value}
                FILTER records._key == @recordId
                RETURN {{
                    type: 'GROUP',
                    source: group,
                    role: permEdge.role
                }}
            )
            LET orgAccess = (
                FOR org, belongsEdge IN 1..1 ANY userDoc._id {CollectionNames.BELONGS_TO.value}
                FILTER belongsEdge.entityType == 'ORGANIZATION'
                FOR records, permEdge IN 1..1 ANY org._id {CollectionNames.PERMISSIONS.value}
                FILTER records._key == @recordId
                RETURN {{
                    type: 'ORGANIZATION',
                    source: org,
                    role: permEdge.role
                }}
            )
            LET kbAccess = kb ? (
                FOR permEdge IN @@permissions_to_kb
                    FILTER permEdge._from == userDoc._id AND permEdge._to == kb._id
                    LIMIT 1
                    LET parentFolder = FIRST(
                        FOR parent, relEdge IN 1..1 INBOUND recordDoc._id @@record_relations
                            FILTER relEdge.relationshipType == 'PARENT_CHILD'
                            FILTER PARSE_IDENTIFIER(parent._id).collection == @files
                            RETURN parent
                    )
                    RETURN {{
                        type: 'KNOWLEDGE_BASE',
                        source: kb,
                        role: permEdge.role,
                        folder: parentFolder
                    }}
            ) : []
            LET anyoneAccess = (
                FOR records IN @@anyone
                FILTER records.organization == @orgId
                    AND records.file_key == @recordId
                RETURN {{
                    type: 'ANYONE',
                    source: null,
                    role: records.role
                }}
            )
            LET allAccess = UNION_DISTINCT(
                directAccess,
                groupAccess,
                orgAccess,
                kbAccess,
                anyoneAccess
            )
            RETURN LENGTH(allAccess) > 0 ? allAccess : null
            """

            bind_vars = {
                "userId": user_id,
                "orgId": org_id,
                "recordId": record_id,
                "@users": CollectionNames.USERS.value,
                "records": CollectionNames.RECORDS.value,
                "files": CollectionNames.FILES.value,
                "@anyone": CollectionNames.ANYONE.value,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@record_relations": CollectionNames.RECORD_RELATIONS.value,
            }

            cursor = self.db.aql.execute(access_query, bind_vars=bind_vars)
            access_result = next(cursor, None)

            if not access_result:
                return None

            # If we have access, get the complete record details
            record = await self.get_document(record_id, CollectionNames.RECORDS.value)
            if not record:
                return None

            user = await self.get_user_by_user_id(user_id)

            # Get file or mail details based on record type
            additional_data = None
            if record["recordType"] == RecordTypes.FILE.value:
                additional_data = await self.get_document(
                    record_id, CollectionNames.FILES.value
                )
            elif record["recordType"] == RecordTypes.MAIL.value:
                additional_data = await self.get_document(
                    record_id, CollectionNames.MAILS.value
                )
                message_id = record["externalRecordId"]
                # Format the webUrl with the user's email
                additional_data["webUrl"] = (
                    f"https://mail.google.com/mail?authuser={user['email']}#all/{message_id}"
                )

            metadata_query = f"""
            LET record = DOCUMENT(CONCAT('{CollectionNames.RECORDS.value}/', @recordId))

            LET departments = (
                FOR dept IN OUTBOUND record._id {CollectionNames.BELONGS_TO_DEPARTMENT.value}
                RETURN {{
                    id: dept._key,
                    name: dept.departmentName
                }}
            )

            LET categories = (
                FOR cat IN OUTBOUND record._id {CollectionNames.BELONGS_TO_CATEGORY.value}
                FILTER PARSE_IDENTIFIER(cat._id).collection == '{CollectionNames.CATEGORIES.value}'
                RETURN {{
                    id: cat._key,
                    name: cat.name
                }}
            )

            LET subcategories1 = (
                FOR subcat IN OUTBOUND record._id {CollectionNames.BELONGS_TO_CATEGORY.value}
                FILTER PARSE_IDENTIFIER(subcat._id).collection == '{CollectionNames.SUBCATEGORIES1.value}'
                RETURN {{
                    id: subcat._key,
                    name: subcat.name
                }}
            )

            LET subcategories2 = (
                FOR subcat IN OUTBOUND record._id {CollectionNames.BELONGS_TO_CATEGORY.value}
                FILTER PARSE_IDENTIFIER(subcat._id).collection == '{CollectionNames.SUBCATEGORIES2.value}'
                RETURN {{
                    id: subcat._key,
                    name: subcat.name
                }}
            )

            LET subcategories3 = (
                FOR subcat IN OUTBOUND record._id {CollectionNames.BELONGS_TO_CATEGORY.value}
                FILTER PARSE_IDENTIFIER(subcat._id).collection == '{CollectionNames.SUBCATEGORIES3.value}'
                RETURN {{
                    id: subcat._key,
                    name: subcat.name
                }}
            )

            LET topics = (
                FOR topic IN OUTBOUND record._id {CollectionNames.BELONGS_TO_TOPIC.value}
                RETURN {{
                    id: topic._key,
                    name: topic.name
                }}
            )

            LET languages = (
                FOR lang IN OUTBOUND record._id {CollectionNames.BELONGS_TO_LANGUAGE.value}
                RETURN {{
                    id: lang._key,
                    name: lang.name
                }}
            )

            RETURN {{
                departments: departments,
                categories: categories,
                subcategories1: subcategories1,
                subcategories2: subcategories2,
                subcategories3: subcategories3,
                topics: topics,
                languages: languages
            }}
            """
            metadata_cursor = self.db.aql.execute(
                metadata_query, bind_vars={"recordId": record_id}
            )
            metadata_result = next(metadata_cursor, None)

            # Get knowledge base info if record is in a KB
            kb_info = None
            folder_info = None
            for access in access_result:
                if access["type"] == "KNOWLEDGE_BASE":
                    kb = access["source"]
                    kb_info = {
                        "id": kb["_key"],
                        "name": kb.get("groupName"),
                        "orgId": kb["orgId"],
                    }
                    if access.get("folder"):
                        folder = access["folder"]
                        folder_info = {
                            "id": folder["_key"],
                            "name": folder["name"]
                        }
                    break

            # Format permissions from access paths
            permissions = []
            for access in access_result:
                permission = {
                    "id": record["_key"],
                    "name": record["recordName"],
                    "type": record["recordType"],
                    "relationship": access["role"],
                    "accessType": access["type"],
                }
                permissions.append(permission)

            return {
                "record": {
                    **record,
                    "fileRecord": (
                        additional_data
                        if record["recordType"] == RecordTypes.FILE.value
                        else None
                    ),
                    "mailRecord": (
                        additional_data
                        if record["recordType"] == RecordTypes.MAIL.value
                        else None
                    ),
                },
                "knowledgeBase": kb_info,
                "folder": folder_info,
                "metadata": metadata_result,
                "permissions": permissions,
            }

        except Exception as e:
            self.logger.error(
                f"Failed to check record access and get details: {str(e)}"
            )
            raise

    async def get_records_by_virtual_record_id(
        self,
        virtual_record_id: str,
        accessible_record_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get all record keys that have the given virtualRecordId.
        Optionally filter by a list of record IDs.

        Args:
            virtual_record_id (str): Virtual record ID to look up
            record_ids (Optional[List[str]]): Optional list of record IDs to filter by

        Returns:
            List[str]: List of record keys that match the criteria
        """
        try:
            self.logger.info(
                "🔍 Finding records with virtualRecordId: %s", virtual_record_id
            )

            # Base query
            query = f"""
            FOR record IN {CollectionNames.RECORDS.value}
                FILTER record.virtualRecordId == @virtual_record_id
            """

            # Add optional filter for record IDs
            if accessible_record_ids:
                query += """
                AND record._key IN @accessible_record_ids
                """

            query += """
                RETURN record._key
            """

            bind_vars = {"virtual_record_id": virtual_record_id}
            if accessible_record_ids:
                bind_vars["accessible_record_ids"] = accessible_record_ids

            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            results = list(cursor)

            self.logger.info(
                "✅ Found %d records with virtualRecordId %s",
                len(results),
                virtual_record_id
            )
            return results

        except Exception as e:
            self.logger.error(
                "❌ Error finding records with virtualRecordId %s: %s",
                virtual_record_id,
                str(e)
            )
            return []

    async def get_records(
        self,
        user_id: str,
        org_id: str,
        skip: int,
        limit: int,
        search: Optional[str],
        record_types: Optional[List[str]],
        origins: Optional[List[str]],
        connectors: Optional[List[str]],
        indexing_status: Optional[List[str]],
        permissions: Optional[List[str]],
        date_from: Optional[int],
        date_to: Optional[int],
        sort_by: str,
        sort_order: str,
        source: str,
    ) -> Tuple[List[Dict], int, Dict]:
        """
        List all records the user can access directly via belongs_to_kb edges.
        Returns (records, total_count, available_filters)
        """
        try:
            self.logger.info(f"🔍 Listing all records for user {user_id}, source: {source}")

            # Determine what data sources to include
            include_kb_records = source in ['all', 'local']
            include_connector_records = source in ['all', 'connector']

            # Build filter conditions function
            def build_record_filters(include_filter_vars: bool = True) -> str:
                conditions = []
                if search and include_filter_vars:
                    conditions.append("(LIKE(LOWER(record.recordName), @search) OR LIKE(LOWER(record.externalRecordId), @search))")
                if record_types and include_filter_vars:
                    conditions.append("record.recordType IN @record_types")
                if origins and include_filter_vars:
                    conditions.append("record.origin IN @origins")
                if connectors and include_filter_vars:
                    conditions.append("record.connectorName IN @connectors")
                if indexing_status and include_filter_vars:
                    conditions.append("record.indexingStatus IN @indexing_status")
                if date_from and include_filter_vars:
                    conditions.append("record.createdAtTimestamp >= @date_from")
                if date_to and include_filter_vars:
                    conditions.append("record.createdAtTimestamp <= @date_to")

                return " AND " + " AND ".join(conditions) if conditions else ""

            base_kb_roles = {"OWNER", "READER", "FILEORGANIZER", "WRITER", "COMMENTER", "ORGANIZER"}
            if permissions:
                # This ensures we only filter by roles that are valid for KBs AND requested by the user.
                final_kb_roles = list(base_kb_roles.intersection(set(permissions)))
                # If the intersection is empty, no KB records will match, which is correct.
                if not final_kb_roles:
                    # To prevent an empty `IN []` which can be inefficient, we can just disable the kbRecords part.
                    include_kb_records = False
            else:
                final_kb_roles = list(base_kb_roles)

            # Build permission filter for connector records
            def build_permission_filter(include_filter_vars: bool = True) -> str:
                if permissions and include_filter_vars:
                    return " AND permissionEdge.role IN @permissions"
                return ""

            # ===== MAIN QUERY (with pagination and filters and file records) =====
            record_filter = build_record_filters(True)
            permission_filter = build_permission_filter(True)

            main_query = f"""
            LET user_from = @user_from
            LET org_id = @org_id
            // KB Records Section - Get records DIRECTLY from belongs_to_kb edges (not through folders)
            LET kbRecords = {
                f'''(
                    FOR kbEdge IN @@permissions_to_kb
                        FILTER kbEdge._from == user_from
                        FILTER kbEdge.type == "USER"
                        FILTER kbEdge.role IN @kb_permissions
                        LET kb = DOCUMENT(kbEdge._to)
                        FILTER kb != null AND kb.orgId == org_id
                        // Get records that belong directly to the KB
                        FOR belongsEdge IN @@belongs_to_kb
                            FILTER belongsEdge._to == kb._id
                            LET record = DOCUMENT(belongsEdge._from)
                            FILTER record != null
                            FILTER record.isDeleted != true
                            FILTER record.orgId == org_id
                            FILTER record.origin == "UPLOAD"
                            // Only include actual records (not folders)
                            FILTER record.isFile != false
                            {record_filter}
                            RETURN {{
                                record: record,
                                permission: {{ role: kbEdge.role, type: kbEdge.type }},
                                kb_id: kb._key,
                                kb_name: kb.groupName
                            }}
                )''' if include_kb_records else '[]'
            }
            // Connector Records Section - Direct connector permissions (FIXED: _to == user_from)
            LET connectorRecords = {
                f'''(
                    FOR permissionEdge IN @@permissions
                        FILTER permissionEdge._to == user_from
                        FILTER permissionEdge.type == "USER"
                        {permission_filter}
                        LET record = DOCUMENT(permissionEdge._from)
                        FILTER record != null
                        FILTER record.recordType != "DRIVE"   //remove this when drive is handled
                        FILTER record.isDeleted != true
                        FILTER record.orgId == org_id
                        FILTER record.origin == "CONNECTOR"
                        {record_filter}
                        RETURN {{
                            record: record,
                            permission: {{ role: permissionEdge.role, type: permissionEdge.type }}
                        }}
                )''' if include_connector_records else '[]'
            }
            LET allRecords = APPEND(kbRecords, connectorRecords)
            FOR item IN allRecords
                LET record = item.record
                SORT record.{sort_by} {sort_order.upper()}
                LIMIT @skip, @limit
                LET fileRecord = FIRST(
                    FOR fileEdge IN @@is_of_type
                        FILTER fileEdge._from == record._id
                        LET file = DOCUMENT(fileEdge._to)
                        FILTER file != null
                        RETURN {{
                            id: file._key,
                            name: file.name,
                            extension: file.extension,
                            mimeType: file.mimeType,
                            sizeInBytes: file.sizeInBytes,
                            isFile: file.isFile,
                            webUrl: file.webUrl
                        }}
                )
                RETURN {{
                    id: record._key,
                    externalRecordId: record.externalRecordId,
                    externalRevisionId: record.externalRevisionId,
                    recordName: record.recordName,
                    recordType: record.recordType,
                    origin: record.origin,
                    connectorName: record.connectorName || "KNOWLEDGE_BASE",
                    indexingStatus: record.indexingStatus,
                    createdAtTimestamp: record.createdAtTimestamp,
                    updatedAtTimestamp: record.updatedAtTimestamp,
                    sourceCreatedAtTimestamp: record.sourceCreatedAtTimestamp,
                    sourceLastModifiedTimestamp: record.sourceLastModifiedTimestamp,
                    orgId: record.orgId,
                    version: record.version,
                    isDeleted: record.isDeleted,
                    deletedByUserId: record.deletedByUserId,
                    isLatestVersion: record.isLatestVersion != null ? record.isLatestVersion : true,
                    webUrl: record.webUrl,
                    fileRecord: fileRecord,
                    permission: {{role: item.permission.role, type: item.permission.type}},
                    kb: {{id: item.kb_id || null, name: item.kb_name || null }}
                }}
            """

            # ===== COUNT QUERY (FIXED: _to == user_from for connector records) =====
            count_query = f"""
            LET user_from = @user_from
            LET org_id = @org_id
            LET kbCount = {
                f'''LENGTH(
                    FOR kbEdge IN @@permissions_to_kb
                        FILTER kbEdge._from == user_from
                        FILTER kbEdge.type == "USER"
                        FILTER kbEdge.role IN @kb_permissions
                        LET kb = DOCUMENT(kbEdge._to)
                        FILTER kb != null AND kb.orgId == org_id
                        FOR belongsEdge IN @@belongs_to_kb
                            FILTER belongsEdge._to == kb._id
                            LET record = DOCUMENT(belongsEdge._from)
                            FILTER record != null
                            FILTER record.isDeleted != true
                            FILTER record.orgId == org_id
                            FILTER record.origin == "UPLOAD"
                            FILTER record.isFile != false
                            {record_filter}
                            RETURN 1
                )''' if include_kb_records else '0'
            }
            LET connectorCount = {
                f'''LENGTH(
                    FOR permissionEdge IN @@permissions
                        FILTER permissionEdge._to == user_from
                        FILTER permissionEdge.type == "USER"
                        {permission_filter}
                        LET record = DOCUMENT(permissionEdge._from)
                        FILTER record != null
                        FILTER record.isDeleted != true
                        FILTER record.orgId == org_id
                        FILTER record.origin == "CONNECTOR"
                        {record_filter}
                        RETURN 1
                )''' if include_connector_records else '0'
            }
            RETURN kbCount + connectorCount
            """

            # ===== FILTERS QUERY (FIXED: _to == user_from for connector records) =====
            filters_query = f"""
            LET user_from = @user_from
            LET org_id = @org_id
            LET allKbRecords = {
                '''(
                    FOR kbEdge IN @@permissions_to_kb
                        FILTER kbEdge._from == user_from
                        FILTER kbEdge.type == "USER"
                        FILTER kbEdge.role IN ["OWNER", "READER", "FILEORGANIZER", "WRITER", "COMMENTER", "ORGANIZER"]
                        LET kb = DOCUMENT(kbEdge._to)
                        FILTER kb != null AND kb.orgId == org_id
                        FOR belongsEdge IN @@belongs_to_kb
                            FILTER belongsEdge._to == kb._id
                            LET record = DOCUMENT(belongsEdge._from)
                            FILTER record != null
                            FILTER record.isDeleted != true
                            FILTER record.orgId == org_id
                            FILTER record.origin == "UPLOAD"
                            FILTER record.isFile != false
                            RETURN {
                                record: record,
                                permission: { role: kbEdge.role }
                            }
                )''' if include_kb_records else '[]'
            }
            LET allConnectorRecords = {
                '''(
                    FOR permissionEdge IN @@permissions
                        FILTER permissionEdge._to == user_from
                        FILTER permissionEdge.type == "USER"
                        LET record = DOCUMENT(permissionEdge._from)
                        FILTER record != null
                        FILTER record.isDeleted != true
                        FILTER record.orgId == org_id
                        FILTER record.origin == "CONNECTOR"
                        RETURN {
                            record: record,
                            permission: { role: permissionEdge.role }
                        }
                )''' if include_connector_records else '[]'
            }
            LET allRecords = APPEND(allKbRecords, allConnectorRecords)
            LET flatRecords = (
                FOR item IN allRecords
                    RETURN item.record
            )
            LET permissionValues = (
                FOR item IN allRecords
                    FILTER item.permission != null
                    RETURN item.permission.role
            )
            LET connectorValues = (
                FOR record IN flatRecords
                    FILTER record.connectorName != null
                    RETURN record.connectorName
            )
            RETURN {{
                recordTypes: UNIQUE(flatRecords[*].recordType) || [],
                origins: UNIQUE(flatRecords[*].origin) || [],
                connectors: UNIQUE(connectorValues) || [],
                indexingStatus: UNIQUE(flatRecords[*].indexingStatus) || [],
                permissions: UNIQUE(permissionValues) || []
            }}
            """

            # Build bind variables
            filter_bind_vars = {}
            if search:
                filter_bind_vars["search"] = f"%{search.lower()}%"
            if record_types:
                filter_bind_vars["record_types"] = record_types
            if origins:
                filter_bind_vars["origins"] = origins
            if connectors:
                filter_bind_vars["connectors"] = connectors
            if indexing_status:
                filter_bind_vars["indexing_status"] = indexing_status
            if permissions:
                filter_bind_vars["permissions"] = permissions
            if date_from:
                filter_bind_vars["date_from"] = date_from
            if date_to:
                filter_bind_vars["date_to"] = date_to

            main_bind_vars = {
                "user_from": f"users/{user_id}",
                "org_id": org_id,
                "skip": skip,
                "limit": limit,
                "kb_permissions": final_kb_roles,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
                "@is_of_type": CollectionNames.IS_OF_TYPE.value,
                **filter_bind_vars,
            }

            count_bind_vars = {
                "user_from": f"users/{user_id}",
                "org_id": org_id,
                "kb_permissions": final_kb_roles,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
                **filter_bind_vars,
            }

            filters_bind_vars = {
                "user_from": f"users/{user_id}",
                "org_id": org_id,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
            }

            # Execute queries
            db = self.db
            records = list(db.aql.execute(main_query, bind_vars=main_bind_vars))
            count = list(db.aql.execute(count_query, bind_vars=count_bind_vars))[0]
            available_filters = list(db.aql.execute(filters_query, bind_vars=filters_bind_vars))[0]

            # Ensure filter structure
            if not available_filters:
                available_filters = {}
            available_filters.setdefault("recordTypes", [])
            available_filters.setdefault("origins", [])
            available_filters.setdefault("connectors", [])
            available_filters.setdefault("indexingStatus", [])
            available_filters.setdefault("permissions", [])

            self.logger.info(f"✅ Listed {len(records)} records out of {count} total")
            return records, count, available_filters

        except Exception as e:
            self.logger.error(f"❌ Failed to list all records: {str(e)}")
            return [], 0, {
                "recordTypes": [],
                "origins": [],
                "connectors": [],
                "indexingStatus": [],
                "permissions": []
            }

    async def get_user_kb_permission(
        self,
        kb_id: str,
        user_id: str,
    ) -> Optional[str]:
        """Validate user knowledge permission"""
        try:
            self.logger.info(f"🔍 Checking permissions for user {user_id} on KB {kb_id}")

            query = """
            FOR perm IN @@permissions_collection
                FILTER perm._from == CONCAT('users/', @user_id)
                FILTER perm._to == CONCAT('recordGroups/', @kb_id)
                RETURN perm
            """

            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    "kb_id": kb_id,
                    "user_id": user_id,
                    "@permissions_collection": CollectionNames.PERMISSIONS_TO_KB.value,
                },
            )

            permission = next(cursor, None)

            if permission:
                role = permission.get("role")
                self.logger.info(f"✅ Found permission: user {user_id} has role '{role}' on KB {kb_id}")
                return role
            else:
                self.logger.warning(f"⚠️ No permission found for user {user_id} on KB {kb_id}")

                # Debug: Let's see what permissions exist for this KB
                debug_query = """
                FOR perm IN @@permissions_collection
                    FILTER perm._to == CONCAT('recordGroups/', @kb_id)
                    RETURN {
                        from: perm._from,
                        role: perm.role,
                        type: perm.type
                    }
                """
                debug_cursor = self.db.aql.execute(
                    debug_query,
                    bind_vars={
                        "kb_id": kb_id,
                        "@permissions_collection": CollectionNames.PERMISSIONS_TO_KB.value,
                    },
                )
                existing_perms = list(debug_cursor)
                self.logger.info(f"🔍 Debug - All permissions for KB {kb_id}: {existing_perms}")

                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to validate knowledge base permission for user {user_id}: {str(e)}")
            raise

    async def reindex_single_record(self, record_id: str, user_id: str, org_id: str) -> Dict:
        """
        Reindex a single record with permission checks and event publishing
        """
        try:
            self.logger.info(f"🔄 Starting reindex for record {record_id} by user {user_id}")

            # Get record to determine connector type
            record = await self.get_document(record_id, CollectionNames.RECORDS.value)
            if not record:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"Record not found: {record_id}"
                }

            if record.get("isDeleted"):
                return {
                    "success": False,
                    "code": 400,
                    "reason": "Cannot reindex deleted record"
                }

            connector_name = record.get("connectorName", "")
            origin = record.get("origin", "")

            self.logger.info(f"📋 Record details - Origin: {origin}, Connector: {connector_name}")

            # Get user
            user = await self.get_user_by_user_id(user_id)
            if not user:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"User not found: {user_id}"
                }

            user_key = user.get('_key')

            # Check permissions based on origin type
            if origin == OriginTypes.UPLOAD.value:
                # KB record - check KB permissions
                kb_context = await self._get_kb_context_for_record(record_id)
                if not kb_context:
                    return {
                        "success": False,
                        "code": 404,
                        "reason": f"Knowledge base context not found for record {record_id}"
                    }

                user_role = await self.get_user_kb_permission(kb_context["kb_id"], user_key)
                if user_role not in ["OWNER", "WRITER"]:
                    return {
                        "success": False,
                        "code": 403,
                        "reason": f"Insufficient KB permissions. User role: {user_role}. Required: OWNER, WRITER"
                    }

                connector_type = Connectors.KNOWLEDGE_BASE.value

            elif origin == OriginTypes.CONNECTOR.value:
                # Connector record - check connector-specific permissions
                if connector_name == Connectors.GOOGLE_DRIVE.value:
                    user_role = await self._check_drive_permissions(record_id, user_key)
                elif connector_name == Connectors.GOOGLE_MAIL.value:
                    user_role = await self._check_gmail_permissions(record_id, user_key)

                if not user_role or user_role not in ["OWNER", "WRITER"]:
                    return {
                        "success": False,
                        "code": 403,
                        "reason": f"Insufficient permissions. User role: {user_role}. Required: OWNER, WRITER"
                    }

                connector_type = connector_name
            else:
                return {
                    "success": False,
                    "code": 400,
                    "reason": f"Unsupported record origin: {origin}"
                }

            # Get file record for event payload
            file_record = await self.get_document(record_id, CollectionNames.FILES.value)

            # Create and publish reindex event
            try:
                payload = await self._create_reindex_event_payload(record, file_record)
                await self._publish_record_event("newRecord",payload)

                self.logger.info(f"✅ Published reindex event for record {record_id}")

                return {
                    "success": True,
                    "recordId": record_id,
                    "recordName": record.get("recordName"),
                    "connector": connector_type,
                    "eventPublished": True,
                    "userRole": user_role
                }

            except Exception as event_error:
                self.logger.error(f"❌ Failed to publish reindex event: {str(event_error)}")
                return {
                    "success": False,
                    "code": 500,
                    "reason": f"Failed to publish reindex event: {str(event_error)}"
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to reindex record {record_id}: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"Internal error: {str(e)}"
            }

    async def reindex_failed_connector_records(self, user_id: str, org_id: str, connector: str, origin: str) -> Dict:
        """
        Reindex all failed records for a specific connector with permission check
        Just validates permissions and publishes a single reindexFailed event
        Args:
            user_id: External user ID doing the reindex
            org_id: Organization ID
            connector: Connector name (GOOGLE_DRIVE, GOOGLE_MAIL, KNOWLEDGE_BASE)
            origin: Origin type (CONNECTOR, UPLOAD)
        Returns:
            Dict: Result with success status and event publication info
        """
        try:
            self.logger.info(f"🔄 Starting failed records reindex for {connector} by user {user_id}")

            # Get user
            user = await self.get_user_by_user_id(user_id)
            if not user:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"User not found: {user_id}"
                }

            user_key = user.get('_key')

            # Check if user has permission to reindex connector records
            permission_check = await self._check_connector_reindex_permissions(
                user_key, org_id, connector, origin
            )

            if not permission_check["allowed"]:
                return {
                    "success": False,
                    "code": 403,
                    "reason": permission_check["reason"]
                }

            # Create and publish single reindexFailed event
            try:
                payload = await self._create_reindex_failed_event_payload(
                    org_id, connector, origin
                )
                await self._publish_sync_event("reindexFailed", payload)

                self.logger.info(f"✅ Published reindexFailed event for {connector}")

                return {
                    "success": True,
                    "connector": connector,
                    "origin": origin,
                    "user_permission_level": permission_check["permission_level"],
                    "event_published": True,
                    "message": f"Successfully initiated reindex of failed {connector} records"
                }

            except Exception as event_error:
                self.logger.error(f"❌ Failed to publish reindexFailed event: {str(event_error)}")
                return {
                    "success": False,
                    "code": 500,
                    "reason": f"Failed to publish reindexFailed event: {str(event_error)}"
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to reindex failed connector records: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"Internal error: {str(e)}"
            }

    async def delete_record(self, record_id: str, user_id: str) -> Dict:
        """
        Main entry point for record deletion - routes to connector-specific methods
        """
        try:
            self.logger.info(f"🚀 Starting record deletion for {record_id} by user {user_id}")

            # Get record to determine connector type
            record = await self.get_document(record_id, CollectionNames.RECORDS.value)
            if not record:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"Record not found: {record_id}"
                }

            connector_name = record.get("connectorName", "")
            origin = record.get("origin", "")

            # Route to connector-specific deletion method
            if origin == OriginTypes.UPLOAD.value or connector_name == Connectors.KNOWLEDGE_BASE.value:
                return await self.delete_knowledge_base_record(record_id, user_id, record)
            elif connector_name == Connectors.GOOGLE_DRIVE.value:
                return await self.delete_google_drive_record(record_id, user_id, record)
            elif connector_name == Connectors.GOOGLE_MAIL.value:
                return await self.delete_gmail_record(record_id, user_id, record)
            else:
                return {
                    "success": False,
                    "code": 400,
                    "reason": f"Unsupported connector: {connector_name}"
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to delete record {record_id}: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"Internal error: {str(e)}"
            }

    async def delete_knowledge_base_record(self, record_id: str, user_id: str, record: Dict) -> Dict:
        """
        Delete a Knowledge Base record - handles uploads and KB-specific logic
        """
        try:
            self.logger.info(f"🗂️ Deleting Knowledge Base record {record_id}")

            # Get user
            user = await self.get_user_by_user_id(user_id)
            if not user:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"User not found: {user_id}"
                }

            user_key = user.get('_key')

            # Find KB context for this record
            kb_context = await self._get_kb_context_for_record(record_id)
            if not kb_context:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"Knowledge base context not found for record {record_id}"
                }

            # Check KB permissions
            user_role = await self.get_user_kb_permission(kb_context["kb_id"], user_key)
            if user_role not in self.connector_delete_permissions[Connectors.KNOWLEDGE_BASE.value]["allowed_roles"]:
                return {
                    "success": False,
                    "code": 403,
                    "reason": f"Insufficient permissions. User role: {user_role}"
                }

            # Execute KB-specific deletion
            return await self._execute_kb_record_deletion(record_id, record, kb_context)

        except Exception as e:
            self.logger.error(f"❌ Failed to delete KB record: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"KB record deletion failed: {str(e)}"
            }

    async def _get_kb_context_for_record(self, record_id: str) -> Optional[Dict]:
        """
        Get KB context for a record
        """
        try:
            self.logger.info(f"🔍 Finding KB context for record {record_id}")

            kb_query = """
            LET record_from = CONCAT('records/', @record_id)
            // Find KB via belongs_to_kb edge
            LET kb_edge = FIRST(
                FOR btk_edge IN @@belongs_to_kb
                    FILTER btk_edge._from == record_from
                    RETURN btk_edge
            )
            LET kb = kb_edge ? DOCUMENT(kb_edge._to) : null
            RETURN kb ? {
                kb_id: kb._key,
                kb_name: kb.groupName,
                org_id: kb.orgId
            } : null
            """

            cursor = self.db.aql.execute(kb_query, bind_vars={
                "record_id": record_id,
                "@belongs_to_kb": CollectionNames.BELONGS_TO_KB.value,
            })

            result = next(cursor, None)

            if result:
                self.logger.info(f"✅ Found KB context: {result['kb_name']}")
                return result
            else:
                self.logger.warning(f"⚠️ No KB context found for record {record_id}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to get KB context for record {record_id}: {str(e)}")
            return None

    async def _execute_kb_record_deletion(self, record_id: str, record: Dict, kb_context: Dict) -> Dict:
        """Execute KB record deletion with transaction"""
        try:
            transaction = self.db.begin_transaction(
                write=self.connector_delete_permissions[Connectors.KNOWLEDGE_BASE.value]["document_collections"] +
                      self.connector_delete_permissions[Connectors.KNOWLEDGE_BASE.value]["edge_collections"]
            )

            try:
                # Get file record for event publishing before deletion
                file_record = await self.get_document(record_id, CollectionNames.FILES.value)

                # Delete KB-specific edges
                await self._delete_kb_specific_edges(transaction, record_id)

                # Delete file record
                if file_record:
                    await self._delete_file_record(transaction, record_id)

                # Delete main record
                await self._delete_main_record(transaction, record_id)

                # Commit transaction
                await asyncio.to_thread(lambda: transaction.commit_transaction())

                # Publish KB deletion event
                try:
                    await self._publish_kb_deletion_event(record, file_record)
                except Exception as event_error:
                    self.logger.error(f"❌ Failed to publish KB deletion event: {str(event_error)}")

                return {
                    "success": True,
                    "record_id": record_id,
                    "connector": Connectors.KNOWLEDGE_BASE.value,
                    "kb_context": kb_context
                }

            except Exception as e:
                await asyncio.to_thread(lambda: transaction.abort_transaction())
                raise e

        except Exception as e:
            self.logger.error(f"❌ KB record deletion transaction failed: {str(e)}")
            return {
                "success": False,
                "reason": f"Transaction failed: {str(e)}"
            }

    async def _delete_kb_specific_edges(self, transaction, record_id: str) -> None:
        """Delete KB-specific edges"""
        kb_edge_collections = self.connector_delete_permissions[Connectors.KNOWLEDGE_BASE.value]["edge_collections"]

        for edge_collection in kb_edge_collections:
            edge_deletion_query = """
            FOR edge IN @@edge_collection
                FILTER edge._from == @record_from OR edge._to == @record_to
                REMOVE edge IN @@edge_collection
                RETURN OLD
            """

            transaction.aql.execute(edge_deletion_query, bind_vars={
                "record_from": f"records/{record_id}",
                "record_to": f"records/{record_id}",
                "@edge_collection": edge_collection,
            })

    async def delete_google_drive_record(self, record_id: str, user_id: str, record: Dict) -> Dict:
        """
        Delete a Google Drive record - handles Drive-specific permissions and logic
        """
        try:
            self.logger.info(f"🔌 Deleting Google Drive record {record_id}")

            # Get user
            user = await self.get_user_by_user_id(user_id)
            if not user:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"User not found: {user_id}"
                }

            user_key = user.get('_key')

            # Check Drive-specific permissions
            user_role = await self._check_drive_permissions(record_id, user_key)
            if not user_role or user_role not in self.connector_delete_permissions[Connectors.GOOGLE_DRIVE.value]["allowed_roles"]:
                return {
                    "success": False,
                    "code": 403,
                    "reason": f"Insufficient Drive permissions. Role: {user_role}"
                }

            # Execute Drive-specific deletion
            return await self._execute_drive_record_deletion(record_id, record, user_role)

        except Exception as e:
            self.logger.error(f"❌ Failed to delete Drive record: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"Drive record deletion failed: {str(e)}"
            }

    async def _execute_drive_record_deletion(self, record_id: str, record: Dict, user_role: str) -> Dict:
        """Execute Drive record deletion with transaction"""
        try:
            transaction = self.db.begin_transaction(
                write=self.connector_delete_permissions[Connectors.GOOGLE_DRIVE.value]["document_collections"] +
                      self.connector_delete_permissions[Connectors.GOOGLE_DRIVE.value]["edge_collections"]
            )

            try:
                # Get file record for event publishing
                file_record = await self.get_document(record_id, CollectionNames.FILES.value)

                # Delete Drive-specific edges
                await self._delete_drive_specific_edges(transaction, record_id)

                # Delete 'anyone' permissions specific to Drive
                await self._delete_drive_anyone_permissions(transaction, record_id)

                # Delete file record
                if file_record:
                    await self._delete_file_record(transaction, record_id)

                # Delete main record
                await self._delete_main_record(transaction, record_id)

                # Commit transaction
                await asyncio.to_thread(lambda: transaction.commit_transaction())

                # Publish Drive deletion event
                try:
                    await self._publish_drive_deletion_event(record, file_record)
                except Exception as event_error:
                    self.logger.error(f"❌ Failed to publish Drive deletion event: {str(event_error)}")

                return {
                    "success": True,
                    "record_id": record_id,
                    "connector": Connectors.GOOGLE_DRIVE.value,
                    "user_role": user_role
                }

            except Exception as e:
                await asyncio.to_thread(lambda: transaction.abort_transaction())
                raise e

        except Exception as e:
            self.logger.error(f"❌ Drive record deletion transaction failed: {str(e)}")
            return {
                "success": False,
                "reason": f"Transaction failed: {str(e)}"
            }

    async def _delete_drive_specific_edges(self, transaction, record_id: str) -> None:
        """Delete Google Drive specific edges with optimized queries"""
        drive_edge_collections = self.connector_delete_permissions[Connectors.GOOGLE_DRIVE.value]["edge_collections"]

        # Define edge deletion strategies - maps collection to query config
        edge_deletion_strategies = {
            CollectionNames.USER_DRIVE_RELATION.value: {
                "filter": "edge._to == CONCAT('drives/', @record_id)",
                "bind_vars": {"record_id": record_id},
                "description": "Drive user relations"
            },
            CollectionNames.IS_OF_TYPE.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "IS_OF_TYPE edges"
            },
            CollectionNames.PERMISSIONS.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "Permission edges"
            },
            CollectionNames.BELONGS_TO.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "Belongs to edges"
            },
            # Default strategy for bidirectional edges
            "default": {
                "filter": "edge._from == @record_from OR edge._to == @record_to",
                "bind_vars": {
                    "record_from": f"records/{record_id}",
                    "record_to": f"records/{record_id}"
                },
                "description": "Bidirectional edges"
            }
        }

        # Single query template for all edge collections
        deletion_query_template = """
        FOR edge IN @@edge_collection
            FILTER {filter}
            REMOVE edge IN @@edge_collection
            RETURN OLD
        """

        total_deleted = 0

        for edge_collection in drive_edge_collections:
            try:
                # Get strategy for this collection or use default
                strategy = edge_deletion_strategies.get(edge_collection, edge_deletion_strategies["default"])

                # Build query with specific filter
                deletion_query = deletion_query_template.format(filter=strategy["filter"])

                # Prepare bind variables
                bind_vars = {
                    "@edge_collection": edge_collection,
                    **strategy["bind_vars"]
                }

                self.logger.debug(f"🔍 Deleting {strategy['description']} from {edge_collection}")
                self.logger.debug(f"🔍 Bind vars: {bind_vars}")

                # Execute deletion
                result = transaction.aql.execute(deletion_query, bind_vars=bind_vars)
                deleted_count = len(list(result))
                total_deleted += deleted_count

                if deleted_count > 0:
                    self.logger.info(f"🗑️ Deleted {deleted_count} {strategy['description']} from {edge_collection}")
                else:
                    self.logger.debug(f"📝 No {strategy['description']} found in {edge_collection}")

            except Exception as e:
                self.logger.error(f"❌ Failed to delete edges from {edge_collection}: {str(e)}")
                self.logger.error(f"❌ Strategy: {strategy}")
                self.logger.error(f"❌ Bind vars: {bind_vars}")
                raise

        self.logger.info(f"✅ Drive edge deletion completed: {total_deleted} total edges deleted for record {record_id}")


    async def _delete_drive_anyone_permissions(self, transaction, record_id: str) -> None:
        """Delete Drive-specific 'anyone' permissions"""
        anyone_deletion_query = """
        FOR anyone_perm IN @@anyone
            FILTER anyone_perm.file_key == @record_id
            REMOVE anyone_perm IN @@anyone
            RETURN OLD
        """

        transaction.aql.execute(anyone_deletion_query, bind_vars={
            "record_id": record_id,
            "@anyone": CollectionNames.ANYONE.value,
        })

    async def delete_gmail_record(self, record_id: str, user_id: str, record: Dict) -> Dict:
        """
        Delete a Gmail record - handles Gmail-specific permissions and logic
        """
        try:
            self.logger.info(f"📧 Deleting Gmail record {record_id}")

            # Get user
            user = await self.get_user_by_user_id(user_id)
            if not user:
                return {
                    "success": False,
                    "code": 404,
                    "reason": f"User not found: {user_id}"
                }

            user_key = user.get('_key')

            # Check Gmail-specific permissions
            user_role = await self._check_gmail_permissions(record_id, user_key)
            if not user_role or user_role not in self.connector_delete_permissions[Connectors.GOOGLE_MAIL.value]["allowed_roles"]:
                return {
                    "success": False,
                    "code": 403,
                    "reason": f"Insufficient Gmail permissions. Role: {user_role}"
                }

            # Execute Gmail-specific deletion
            return await self._execute_gmail_record_deletion(record_id, record, user_role)

        except Exception as e:
            self.logger.error(f"❌ Failed to delete Gmail record: {str(e)}")
            return {
                "success": False,
                "code": 500,
                "reason": f"Gmail record deletion failed: {str(e)}"
            }

    async def _execute_gmail_record_deletion(self, record_id: str, record: Dict, user_role: str) -> Dict:
        """Execute Gmail record deletion with transaction"""
        try:
            transaction = self.db.begin_transaction(
                write=self.connector_delete_permissions[Connectors.GOOGLE_MAIL.value]["document_collections"] +
                      self.connector_delete_permissions[Connectors.GOOGLE_MAIL.value]["edge_collections"]
            )

            try:
                # Get mail and file records for event publishing
                mail_record = await self.get_document(record_id, CollectionNames.MAILS.value)
                file_record = await self.get_document(record_id, CollectionNames.FILES.value) if record.get("recordType") == "FILE" else None

                # Delete Gmail-specific edges (including thread relationships)
                await self._delete_gmail_specific_edges(transaction, record_id)

                # Delete mail record
                if mail_record:
                    await self._delete_mail_record(transaction, record_id)

                # Delete file record if it's an attachment
                if file_record:
                    await self._delete_file_record(transaction, record_id)

                # Delete main record
                await self._delete_main_record(transaction, record_id)

                # Commit transaction
                await asyncio.to_thread(lambda: transaction.commit_transaction())

                # Publish Gmail deletion event
                try:
                    await self._publish_gmail_deletion_event(record, mail_record, file_record)
                except Exception as event_error:
                    self.logger.error(f"❌ Failed to publish Gmail deletion event: {str(event_error)}")

                return {
                    "success": True,
                    "record_id": record_id,
                    "connector": Connectors.GOOGLE_MAIL.value,
                    "user_role": user_role
                }

            except Exception as e:
                await asyncio.to_thread(lambda: transaction.abort_transaction())
                raise e

        except Exception as e:
            self.logger.error(f"❌ Gmail record deletion transaction failed: {str(e)}")
            return {
                "success": False,
                "reason": f"Transaction failed: {str(e)}"
            }

    async def _delete_gmail_specific_edges(self, transaction, record_id: str) -> None:
        """Delete Gmail specific edges with optimized queries"""
        gmail_edge_collections = self.connector_delete_permissions[Connectors.GOOGLE_MAIL.value]["edge_collections"]

        # Define edge deletion strategies - maps collection to query config
        edge_deletion_strategies = {
            CollectionNames.IS_OF_TYPE.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "IS_OF_TYPE edges"
            },
            CollectionNames.RECORD_RELATIONS.value: {
                "filter": "(edge._from == @record_from OR edge._to == @record_to) AND edge.relationType IN @relation_types",
                "bind_vars": {
                    "record_from": f"records/{record_id}",
                    "record_to": f"records/{record_id}",
                    "relation_types": ["SIBLING", "ATTACHMENT"]  # Gmail-specific relation types
                },
                "description": "Gmail record relations (SIBLING/ATTACHMENT)"
            },
            CollectionNames.PERMISSIONS.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "Permission edges"
            },
            CollectionNames.BELONGS_TO.value: {
                "filter": "edge._from == @record_from",
                "bind_vars": {"record_from": f"records/{record_id}"},
                "description": "Belongs to edges"
            },
            # Default strategy for any other collections
            "default": {
                "filter": "edge._from == @record_from OR edge._to == @record_to",
                "bind_vars": {
                    "record_from": f"records/{record_id}",
                    "record_to": f"records/{record_id}"
                },
                "description": "Bidirectional edges"
            }
        }

        # Single query template for all edge collections
        deletion_query_template = """
        FOR edge IN @@edge_collection
            FILTER {filter}
            REMOVE edge IN @@edge_collection
            RETURN OLD
        """

        total_deleted = 0

        for edge_collection in gmail_edge_collections:
            try:
                # Get strategy for this collection or use default
                strategy = edge_deletion_strategies.get(edge_collection, edge_deletion_strategies["default"])

                # Build query with specific filter
                deletion_query = deletion_query_template.format(filter=strategy["filter"])

                # Prepare bind variables
                bind_vars = {
                    "@edge_collection": edge_collection,
                    **strategy["bind_vars"]
                }

                self.logger.debug(f"🔍 Deleting {strategy['description']} from {edge_collection}")
                self.logger.debug(f"🔍 Bind vars: {bind_vars}")

                # Execute deletion
                result = transaction.aql.execute(deletion_query, bind_vars=bind_vars)
                deleted_count = len(list(result))
                total_deleted += deleted_count

                if deleted_count > 0:
                    self.logger.info(f"🗑️ Deleted {deleted_count} {strategy['description']} from {edge_collection}")
                else:
                    self.logger.debug(f"📝 No {strategy['description']} found in {edge_collection}")

            except Exception as e:
                self.logger.error(f"❌ Failed to delete edges from {edge_collection}: {str(e)}")
                self.logger.error(f"❌ Strategy: {strategy}")
                self.logger.error(f"❌ Bind vars: {bind_vars}")
                raise

        self.logger.info(f"✅ Gmail edge deletion completed: {total_deleted} total edges deleted for record {record_id}")

    async def _delete_file_record(self, transaction, record_id: str) -> None:
        """Delete file record from files collection"""
        file_deletion_query = """
        REMOVE @record_id IN @@files_collection
        RETURN OLD
        """

        transaction.aql.execute(file_deletion_query, bind_vars={
            "record_id": record_id,
            "@files_collection": CollectionNames.FILES.value,
        })

    async def _delete_mail_record(self, transaction, record_id: str) -> None:
        """Delete mail record from mails collection"""
        mail_deletion_query = """
        REMOVE @record_id IN @@mails_collection
        RETURN OLD
        """

        transaction.aql.execute(mail_deletion_query, bind_vars={
            "record_id": record_id,
            "@mails_collection": CollectionNames.MAILS.value,
        })

    async def _delete_main_record(self, transaction, record_id: str) -> None:
        """Delete main record from records collection"""
        record_deletion_query = """
        REMOVE @record_id IN @@records_collection
        RETURN OLD
        """

        transaction.aql.execute(record_deletion_query, bind_vars={
            "record_id": record_id,
            "@records_collection": CollectionNames.RECORDS.value,
        })

    async def _check_connector_reindex_permissions(self, user_key: str, org_id: str, connector: str, origin: str) -> Dict:
        """
        Simple permission check for connector reindex operations
        Permission rules:
        1. Organization OWNER - Can reindex any connector
        2. Knowledge Base OWNER - Can reindex KB records only
        3. Users with significant connector access (≥50% of records)
        """
        try:
            self.logger.info(f"🔍 Checking connector reindex permissions for user {user_key}")

            permission_query = """
            LET user = DOCUMENT("users", @user_key)
            FILTER user != null
            // Check organization ownership
            LET org_owner = FIRST(
                FOR edge IN @@belongs_to
                    FILTER edge._from == user._id
                    FILTER edge._to == CONCAT('organizations/', @org_id)
                    FILTER edge.entityType == 'ORGANIZATION'
                    FILTER edge.role == 'OWNER'
                    RETURN edge
            )
            // Check if user is KB owner (for KB connectors)
            LET kb_owner_count = @origin == 'UPLOAD' ? (
                LENGTH(
                    FOR perm IN @@permissions_to_kb
                        FILTER perm._from == user._id
                        FILTER perm.role == 'OWNER'
                        LET kb = DOCUMENT(perm._to)
                        FILTER kb != null AND kb.orgId == @org_id
                        RETURN perm
                )
            ) : 0
            // Check connector-specific permissions (simplified count)
            LET connector_access_count = @origin == 'CONNECTOR' ? (
                LENGTH(
                    FOR perm IN @@permissions
                        FILTER perm._to == user._id
                        FILTER perm.role IN ['OWNER', 'WRITER']
                        LET record = DOCUMENT(perm._from)
                        FILTER record != null
                        FILTER record.orgId == @org_id
                        FILTER record.connectorName == @connector
                        FILTER record.origin == @origin
                        RETURN perm
                )
            ) : 0
            // Get total connector records for percentage calculation
            LET total_connector_records = @origin == 'CONNECTOR' ? (
                LENGTH(
                    FOR record IN @@records
                        FILTER record.orgId == @org_id
                        FILTER record.connectorName == @connector
                        FILTER record.origin == @origin
                        FILTER record.isDeleted != true
                        RETURN record
                )
            ) : (
                LENGTH(
                    FOR record IN @@records
                        FILTER record.orgId == @org_id
                        FILTER record.origin == @origin
                        FILTER record.isDeleted != true
                        RETURN record
                )
            )
            LET access_percentage = total_connector_records > 0 ?
                (connector_access_count * 100.0 / total_connector_records) : 0
            // Determine permission level
            LET permission_level = (
                org_owner ? 'ORGANIZATION_OWNER' :
                kb_owner_count > 0 ? 'KB_OWNER' :
                access_percentage >= 50 ? 'SUFFICIENT_ACCESS' :
                'INSUFFICIENT_ACCESS'
            )
            // Simple permission logic
            LET allowed = (
                org_owner != null OR
                (kb_owner_count > 0 AND @origin == 'UPLOAD') OR
                (@origin == 'CONNECTOR' AND access_percentage >= 50)
            )
            RETURN {
                allowed: allowed,
                permission_level: permission_level,
                access_percentage: access_percentage,
                total_records: total_connector_records,
                accessible_records: connector_access_count,
                reason: !allowed ? (
                    @origin == 'UPLOAD' ? 'User must be a Knowledge Base owner to reindex KB records' :
                    access_percentage < 50 ? 'User has insufficient access to connector records (less than 50%)' :
                    'User has no permission to reindex connector records'
                ) : 'Permission granted'
            }
            """

            cursor = self.db.aql.execute(permission_query, bind_vars={
                "user_key": user_key,
                "org_id": org_id,
                "connector": connector,
                "origin": origin,
                "@belongs_to": CollectionNames.BELONGS_TO.value,
                "@permissions_to_kb": CollectionNames.PERMISSIONS_TO_KB.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@records": CollectionNames.RECORDS.value,
            })

            result = next(cursor, {})

            if result.get("allowed"):
                self.logger.info(f"✅ Permission granted for connector reindex: {result['permission_level']}")
            else:
                self.logger.warning(f"⚠️ Permission denied for connector reindex: {result.get('reason')}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error checking connector reindex permissions: {str(e)}")
            return {
                "allowed": False,
                "reason": f"Permission check failed: {str(e)}",
                "permission_level": "ERROR"
            }

    async def _check_drive_permissions(self, record_id: str, user_key: str) -> Optional[str]:
        """
        Check Google Drive specific permissions
        Checks: Direct permissions, Group permissions, Domain permissions, Anyone permissions, Drive-level access
        """
        try:
            self.logger.info(f"🔍 Checking Drive permissions for record {record_id} and user {user_key}")

            drive_permission_query = """
            LET user_from = CONCAT('users/', @user_key)
            LET record_from = CONCAT('records/', @record_id)
            // 1. Check direct user permissions on the record
            LET direct_permission = FIRST(
                FOR perm IN @@permissions
                    FILTER perm._from == record_from
                    FILTER perm._to == user_from
                    FILTER perm.type == "USER"
                    RETURN perm.role
            )
            // 2. Check group permissions
            LET group_permission = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "GROUP"
                    LET group = DOCUMENT(belongs_edge._to)
                    FILTER group != null
                    FOR perm IN @@permissions
                        FILTER perm._from == record_from
                        FILTER perm._to == group._id
                        FILTER perm.type == "GROUP"
                        RETURN perm.role
            )
            // 3. Check domain/organization permissions
            LET domain_permission = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "ORGANIZATION"
                    LET org = DOCUMENT(belongs_edge._to)
                    FILTER org != null
                    FOR perm IN @@permissions
                        FILTER perm._from == record_from
                        FILTER perm._to == org._id
                        FILTER perm.type == "DOMAIN"
                        RETURN perm.role
            )
            // 4. Check 'anyone' permissions (Drive-specific)
            LET user_org_id = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "ORGANIZATION"
                    LET org = DOCUMENT(belongs_edge._to)
                    FILTER org != null
                    RETURN org._key
            )
            LET anyone_permission = user_org_id ? FIRST(
                FOR anyone_perm IN @@anyone
                    FILTER anyone_perm.file_key == @record_id
                    FILTER anyone_perm.organization == user_org_id
                    FILTER anyone_perm.active == true
                    RETURN anyone_perm.role
            ) : null
            // 5. Check Drive-level access (user-drive relationship)
            LET drive_access = FIRST(
                // Get the file record to find its drive
                FOR record IN @@records
                    FILTER record._key == @record_id
                    FOR file_edge IN @@is_of_type
                        FILTER file_edge._from == record._id
                        LET file = DOCUMENT(file_edge._to)
                        FILTER file != null
                        // Get the drive this file belongs to
                        LET file_drive_id = file.driveId
                        FILTER file_drive_id != null
                        // Check if user has access to this drive
                        FOR drive_edge IN @@user_drive_relation
                            FILTER drive_edge._from == user_from
                            LET drive = DOCUMENT(drive_edge._to)
                            FILTER drive != null
                            FILTER drive._key == file_drive_id OR drive.driveId == file_drive_id
                            // Map drive access level to permission role
                            LET drive_role = (
                                drive_edge.access_level == "owner" ? "OWNER" :
                                drive_edge.access_level IN ["writer", "fileOrganizer"] ? "WRITER" :
                                drive_edge.access_level IN ["commenter", "reader"] ? "READER" :
                                null
                            )
                            RETURN drive_role
            )
            // Return the highest permission level found (in order of precedence)
            LET final_permission = (
                direct_permission ? direct_permission :
                group_permission ? group_permission :
                domain_permission ? domain_permission :
                anyone_permission ? anyone_permission :
                drive_access ? drive_access :
                null
            )
            RETURN {
                permission: final_permission,
                source: (
                    direct_permission ? "DIRECT" :
                    group_permission ? "GROUP" :
                    domain_permission ? "DOMAIN" :
                    anyone_permission ? "ANYONE" :
                    drive_access ? "DRIVE_ACCESS" :
                    "NONE"
                )
            }
            """

            cursor = self.db.aql.execute(drive_permission_query, bind_vars={
                "record_id": record_id,
                "user_key": user_key,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@belongs_to": CollectionNames.BELONGS_TO.value,
                "@anyone": CollectionNames.ANYONE.value,
                "@records": CollectionNames.RECORDS.value,
                "@is_of_type": CollectionNames.IS_OF_TYPE.value,
                "@user_drive_relation": CollectionNames.USER_DRIVE_RELATION.value,
            })

            result = next(cursor, None)

            if result and result.get("permission"):
                permission = result["permission"]
                source = result["source"]
                self.logger.info(f"✅ Drive permission found: {permission} (via {source})")
                return permission
            else:
                self.logger.warning(f"⚠️ No Drive permissions found for user {user_key} on record {record_id}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to check Drive permissions: {str(e)}")
            return None

    async def _check_gmail_permissions(self, record_id: str, user_key: str) -> Optional[str]:
        """
        Check Gmail specific permissions
        Gmail permission model: User must be sender, recipient (to/cc/bcc), or have explicit permissions
        """
        try:
            self.logger.info(f"🔍 Checking Gmail permissions for record {record_id} and user {user_key}")

            gmail_permission_query = """
            LET user_from = CONCAT('users/', @user_key)
            LET record_from = CONCAT('records/', @record_id)
            // Get user details
            LET user = DOCUMENT(user_from)
            LET user_email = user ? user.email : null
            // 1. Check if user is sender/recipient of the email
            LET email_access = user_email ? (
                FOR record IN @@records
                    FILTER record._key == @record_id
                    FILTER record.recordType == "MAIL"
                    // Get the mail record
                    FOR mail_edge IN @@is_of_type
                        FILTER mail_edge._from == record._id
                        LET mail = DOCUMENT(mail_edge._to)
                        FILTER mail != null
                        // Check if user is sender
                        LET is_sender = mail.from == user_email OR mail.senderEmail == user_email
                        // Check if user is in recipients (to, cc, bcc)
                        LET is_in_to = user_email IN (mail.to || [])
                        LET is_in_cc = user_email IN (mail.cc || [])
                        LET is_in_bcc = user_email IN (mail.bcc || [])
                        LET is_recipient = is_in_to OR is_in_cc OR is_in_bcc
                        FILTER is_sender OR is_recipient
                        // Return role based on relationship
                        RETURN is_sender ? "OWNER" : "READER"
            ) : []
            LET email_permission = LENGTH(email_access) > 0 ? FIRST(email_access) : null
            // 2. Check direct user permissions on the record
            LET direct_permission = FIRST(
                FOR perm IN @@permissions
                    FILTER perm._from == record_from
                    FILTER perm._to == user_from
                    FILTER perm.type == "USER"
                    RETURN perm.role
            )
            // 3. Check group permissions
            LET group_permission = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "GROUP"
                    LET group = DOCUMENT(belongs_edge._to)
                    FILTER group != null
                    FOR perm IN @@permissions
                        FILTER perm._from == record_from
                        FILTER perm._to == group._id
                        FILTER perm.type == "GROUP"
                        RETURN perm.role
            )
            // 4. Check domain/organization permissions
            LET domain_permission = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "ORGANIZATION"
                    LET org = DOCUMENT(belongs_edge._to)
                    FILTER org != null
                    FOR perm IN @@permissions
                        FILTER perm._from == record_from
                        FILTER perm._to == org._id
                        FILTER perm.type == "DOMAIN"
                        RETURN perm.role
            )
            // 5. Check 'anyone' permissions
            LET user_org_id = FIRST(
                FOR belongs_edge IN @@belongs_to
                    FILTER belongs_edge._from == user_from
                    FILTER belongs_edge.entityType == "ORGANIZATION"
                    LET org = DOCUMENT(belongs_edge._to)
                    FILTER org != null
                    RETURN org._key
            )
            LET anyone_permission = user_org_id ? FIRST(
                FOR anyone_perm IN @@anyone
                    FILTER anyone_perm.file_key == @record_id
                    FILTER anyone_perm.organization == user_org_id
                    FILTER anyone_perm.active == true
                    RETURN anyone_perm.role
            ) : null
            // Return the highest permission level found (email access takes precedence)
            LET final_permission = (
                email_permission ? email_permission :
                direct_permission ? direct_permission :
                group_permission ? group_permission :
                domain_permission ? domain_permission :
                anyone_permission ? anyone_permission :
                null
            )
            RETURN {
                permission: final_permission,
                source: (
                    email_permission ? "EMAIL_ACCESS" :
                    direct_permission ? "DIRECT" :
                    group_permission ? "GROUP" :
                    domain_permission ? "DOMAIN" :
                    anyone_permission ? "ANYONE" :
                    "NONE"
                ),
                user_email: user_email,
                is_sender: email_permission == "OWNER",
                is_recipient: email_permission == "READER"
            }
            """

            cursor = self.db.aql.execute(gmail_permission_query, bind_vars={
                "record_id": record_id,
                "user_key": user_key,
                "@records": CollectionNames.RECORDS.value,
                "@is_of_type": CollectionNames.IS_OF_TYPE.value,
                "@permissions": CollectionNames.PERMISSIONS.value,
                "@belongs_to": CollectionNames.BELONGS_TO.value,
                "@anyone": CollectionNames.ANYONE.value,
            })

            result = next(cursor, None)

            if result and result.get("permission"):
                permission = result["permission"]
                source = result["source"]
                user_email = result.get("user_email", "unknown")

                if source == "EMAIL_ACCESS":
                    role_type = "sender" if result.get("is_sender") else "recipient"
                    self.logger.info(f"✅ Gmail permission found: {permission} (user {user_email} is {role_type})")
                else:
                    self.logger.info(f"✅ Gmail permission found: {permission} (via {source})")

                return permission
            else:
                self.logger.warning(f"⚠️ No Gmail permissions found for user {user_key} on record {record_id}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to check Gmail permissions: {str(e)}")
            return None

    async def _create_deleted_record_event_payload(
        self,
        record: Dict,
        file_record: Optional[Dict] = None
    ) -> Dict:
        """Create deleted record event payload matching Node.js format"""
        try:
            # Get extension and mimeType from file record
            extension = ""
            mime_type = ""
            if file_record:
                extension = file_record.get("extension", "")
                mime_type = file_record.get("mimeType", "")

            return {
                "orgId": record.get("orgId"),
                "recordId": record.get("_key"),
                "version": record.get("version", 1),
                "extension": extension,
                "mimeType": mime_type,
                "summaryDocumentId": record.get("summaryDocumentId"),
                "virtualRecordId": record.get("virtualRecordId"),
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to create deleted record event payload: {str(e)}")
            return {}

    async def _create_reindex_event_payload(self, record: Dict, file_record: Optional[Dict]) -> Dict:
        """Create reindex event payload"""
        try:
            # Get extension and mimeType from file record
            extension = ""
            mime_type = ""
            if file_record:
                extension = file_record.get("extension", "")
                mime_type = file_record.get("mimeType", "")

            endpoints = await self.config.get_config(
                    config_node_constants.ENDPOINTS.value
                )
            storage_url = endpoints.get("storage").get("endpoint", DefaultEndpoints.STORAGE_ENDPOINT.value)

            signed_url_route = f"{storage_url}/api/v1/document/internal/{record['externalRecordId']}/download"

            return {
                "orgId": record.get("orgId"),
                "recordId": record.get("_key"),
                "recordName": record.get("recordName", ""),
                "recordType": record.get("recordType", ""),
                "version": record.get("version", 1),
                "signedUrlRoute": signed_url_route,
                "origin": record.get("origin", ""),
                "extension": extension,
                "mimeType": mime_type,
                "createdAtTimestamp": str(record.get("createdAtTimestamp", get_epoch_timestamp_in_ms())),
                "updatedAtTimestamp": str(get_epoch_timestamp_in_ms()),
                "sourceCreatedAtTimestamp": str(record.get("sourceCreatedAtTimestamp", record.get("createdAtTimestamp", get_epoch_timestamp_in_ms())))
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to create reindex event payload: {str(e)}")
            raise

    async def _create_reindex_failed_event_payload(self, orgId:str, connector: str, origin: str) -> Dict:
        """Create reindex connector records event payload"""
        try:

            return {
                "orgId": orgId,
                "origin": origin,
                "connector": connector,
                "createdAtTimestamp": str(get_epoch_timestamp_in_ms()),
                "updatedAtTimestamp": str(get_epoch_timestamp_in_ms()),
                "sourceCreatedAtTimestamp": str(get_epoch_timestamp_in_ms())
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to create reindex event payload: {str(e)}")
            raise

    async def _publish_sync_event(self, event_type: str, payload: Dict) -> None:
        """Publish record event to Kafka"""
        try:
            timestamp = get_epoch_timestamp_in_ms()

            event = {
                "eventType": event_type,
                "timestamp": timestamp,
                "payload": payload
            }

            await self.kafka_service.publish_event("sync-events", event)
            self.logger.info(f"✅ Published {event_type} event for record {payload.get('recordId')}")

        except Exception as e:
            self.logger.error(f"❌ Failed to publish {event_type} event: {str(e)}")

    async def _publish_record_event(self, event_type: str, payload: Dict) -> None:
        """Publish record event to Kafka"""
        try:
            timestamp = get_epoch_timestamp_in_ms()

            event = {
                "eventType": event_type,
                "timestamp": timestamp,
                "payload": payload
            }

            await self.kafka_service.publish_event("record-events", event)
            self.logger.info(f"✅ Published {event_type} event for record {payload.get('recordId')}")

        except Exception as e:
            self.logger.error(f"❌ Failed to publish {event_type} event: {str(e)}")

    async def _publish_kb_deletion_event(self, record: Dict, file_record: Optional[Dict]) -> None:
        """Publish KB-specific deletion event"""
        try:
            payload = await self._create_deleted_record_event_payload(record, file_record)
            if payload:
                # Add KB-specific metadata
                payload["connectorName"] = Connectors.KNOWLEDGE_BASE.value
                payload["origin"] = OriginTypes.UPLOAD.value

                await self._publish_record_event("deleteRecord", payload)
        except Exception as e:
            self.logger.error(f"❌ Failed to publish KB deletion event: {str(e)}")

    async def _publish_drive_deletion_event(self, record: Dict, file_record: Optional[Dict]) -> None:
        """Publish Drive-specific deletion event"""
        try:
            payload = await self._create_deleted_record_event_payload(record, file_record)
            if payload:
                # Add Drive-specific metadata
                payload["connectorName"] = Connectors.GOOGLE_DRIVE.value
                payload["origin"] = OriginTypes.CONNECTOR.value

                # Add Drive-specific fields if available
                if file_record:
                    payload["driveId"] = file_record.get("driveId", "")
                    payload["parentId"] = file_record.get("parentId", "")
                    payload["webViewLink"] = file_record.get("webViewLink", "")

                await self._publish_record_event("deleteRecord", payload)
        except Exception as e:
            self.logger.error(f"❌ Failed to publish Drive deletion event: {str(e)}")

    async def _publish_gmail_deletion_event(self, record: Dict, mail_record: Optional[Dict], file_record: Optional[Dict]) -> None:
        """Publish Gmail-specific deletion event"""
        try:
            # Use mail_record or file_record for attachment info
            data_record = mail_record or file_record
            payload = await self._create_deleted_record_event_payload(record, data_record)

            if payload:
                # Add Gmail-specific metadata
                payload["connectorName"] = Connectors.GOOGLE_MAIL.value
                payload["origin"] = OriginTypes.CONNECTOR.value

                # Add Gmail-specific fields if available
                if mail_record:
                    payload["messageId"] = mail_record.get("messageId", "")
                    payload["threadId"] = mail_record.get("threadId", "")
                    payload["subject"] = mail_record.get("subject", "")
                    payload["from"] = mail_record.get("from", "")
                    payload["isAttachment"] = False
                elif file_record:
                    # This is an email attachment
                    payload["isAttachment"] = True
                    payload["attachmentId"] = file_record.get("attachmentId", "")

                await self._publish_record_event("deleteRecord", payload)
        except Exception as e:
            self.logger.error(f"❌ Failed to publish Gmail deletion event: {str(e)}")

