# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.lenient._Lenient`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from collections.abc import Iterable

from iris.common.lenient import (
    _LENIENT_ENABLE_DEFAULT,
    _LENIENT_PROTECTED,
    _Lenient,
    _qualname,
)


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.expected = dict(active=None, enable=_LENIENT_ENABLE_DEFAULT)

    def test_default(self):
        lenient = _Lenient()
        self.assertEqual(self.expected, lenient.__dict__)

    def test_args_service_str(self):
        service = "service1"
        lenient = _Lenient(service)
        self.expected.update(dict(service1=True))
        self.assertEqual(self.expected, lenient.__dict__)

    def test_args_services_str(self):
        services = ("service1", "service2")
        lenient = _Lenient(*services)
        self.expected.update(dict(service1=True, service2=True))
        self.assertEqual(self.expected, lenient.__dict__)

    def test_args_services_callable(self):
        def service1():
            pass

        def service2():
            pass

        services = (service1, service2)
        lenient = _Lenient(*services)
        self.expected.update({_qualname(service1): True, _qualname(service2): True})
        self.assertEqual(self.expected, lenient.__dict__)

    def test_kwargs_client_str(self):
        client = dict(client1="service1")
        lenient = _Lenient(**client)
        self.expected.update(dict(client1=("service1",)))
        self.assertEqual(self.expected, lenient.__dict__)

    def test_kwargs_clients_str(self):
        clients = dict(client1="service1", client2="service2")
        lenient = _Lenient(**clients)
        self.expected.update(dict(client1=("service1",), client2=("service2",)))
        self.assertEqual(self.expected, lenient.__dict__)

    def test_kwargs_clients_callable(self):
        def client1():
            pass

        def client2():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client1 = _qualname(client1)
        qualname_client2 = _qualname(client2)
        clients = {
            qualname_client1: service1,
            qualname_client2: (service1, service2),
        }
        lenient = _Lenient(**clients)
        self.expected.update(
            {
                _qualname(client1): (_qualname(service1),),
                _qualname(client2): (_qualname(service1), _qualname(service2)),
            }
        )
        self.assertEqual(self.expected, lenient.__dict__)


class Test___call__(tests.IrisTest):
    def setUp(self):
        self.client = "myclient"
        self.lenient = _Lenient()

    def test_missing_service_str(self):
        self.assertFalse(self.lenient("myservice"))

    def test_missing_service_callable(self):
        def myservice():
            pass

        self.assertFalse(self.lenient(myservice))

    def test_disabled_service_str(self):
        service = "myservice"
        self.lenient.__dict__[service] = False
        self.assertFalse(self.lenient(service))

    def test_disable_service_callable(self):
        def myservice():
            pass

        qualname_service = _qualname(myservice)
        self.lenient.__dict__[qualname_service] = False
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_no_active_client(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_no_active_client(self):
        def myservice():
            pass

        qualname_service = _qualname(myservice)
        self.lenient.__dict__[qualname_service] = True
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_no_registered_services(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_no_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = _qualname(myservice)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = _qualname(myclient)
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_unmatched_registered_services(
        self,
    ):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = ("service1", "service2")
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_unmatched_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = _qualname(myservice)
        qualname_client = _qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = ("service1", "service2")
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_registered_services(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = ("service1", "service2", service)
        self.assertTrue(self.lenient(service))

    def test_service_callable_with_active_client_with_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = _qualname(myservice)
        qualname_client = _qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = (
            "service1",
            "service2",
            qualname_service,
        )
        self.assertTrue(self.lenient(myservice))

    def test_service_str_with_active_client_with_unmatched_registered_service_str(
        self,
    ):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = "serviceXXX"
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_unmatched_registered_service_str(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = _qualname(myservice)
        qualname_client = _qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = f"{qualname_service}XXX"
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_registered_service_str(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = service
        self.assertTrue(self.lenient(service))

    def test_service_callable_with_active_client_with_registered_service_str(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = _qualname(myservice)
        qualname_client = _qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = qualname_service
        self.assertTrue(self.lenient(myservice))

    def test_enable(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = service
        self.assertTrue(self.lenient(service))
        self.lenient.__dict__["enable"] = False
        self.assertFalse(self.lenient(service))


class Test___contains__(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_in(self):
        self.assertIn("active", self.lenient)

    def test_not_in(self):
        self.assertNotIn("ACTIVATE", self.lenient)

    def test_in_qualname(self):
        def func():
            pass

        qualname_func = _qualname(func)
        lenient = _Lenient()
        lenient.__dict__[qualname_func] = None
        self.assertIn(func, lenient)
        self.assertIn(qualname_func, lenient)


class Test___getattr__(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_in(self):
        self.assertIsNone(self.lenient.active)

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = self.lenient.wibble


class Test__getitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_in(self):
        self.assertIsNone(self.lenient["active"])

    def test_in_callable(self):
        def service():
            pass

        qualname_service = _qualname(service)
        self.lenient.__dict__[qualname_service] = True
        self.assertTrue(self.lenient[service])

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = self.lenient["wibble"]

    def test_not_in_callable(self):
        def service():
            pass

        qualname_service = _qualname(service)
        emsg = f"Invalid .* option, got '{qualname_service}'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = self.lenient[service]


class Test___setitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(KeyError, emsg):
            self.lenient["wibble"] = None

    def test_in_value_str(self):
        client = "client"
        service = "service"
        self.lenient.__dict__[client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[client], (service,))

    def test_callable_in_value_str(self):
        def client():
            pass

        service = "service"
        qualname_client = _qualname(client)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[qualname_client], (service,))

    def test_in_value_callable(self):
        def service():
            pass

        client = "client"
        qualname_service = _qualname(service)
        self.lenient.__dict__[client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[client], (qualname_service,))

    def test_callable_in_value_callable(self):
        def client():
            pass

        def service():
            pass

        qualname_client = _qualname(client)
        qualname_service = _qualname(service)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[qualname_client], (qualname_service,))

    def test_in_value_bool(self):
        client = "client"
        self.lenient.__dict__[client] = None
        self.lenient[client] = True
        self.assertTrue(self.lenient.__dict__[client])
        self.assertFalse(isinstance(self.lenient.__dict__[client], Iterable))

    def test_callable_in_value_bool(self):
        def client():
            pass

        qualname_client = _qualname(client)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = True
        self.assertTrue(self.lenient.__dict__[qualname_client])
        self.assertFalse(isinstance(self.lenient.__dict__[qualname_client], Iterable))

    def test_in_value_iterable(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.__dict__[client] = None
        self.lenient[client] = services
        self.assertEqual(self.lenient.__dict__[client], services)

    def test_callable_in_value_iterable(self):
        def client():
            pass

        qualname_client = _qualname(client)
        services = ("service1", "service2")
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = services
        self.assertEqual(self.lenient.__dict__[qualname_client], services)

    def test_in_value_iterable_callable(self):
        def service1():
            pass

        def service2():
            pass

        client = "client"
        self.lenient.__dict__[client] = None
        qualname_services = (_qualname(service1), _qualname(service2))
        self.lenient[client] = (service1, service2)
        self.assertEqual(self.lenient.__dict__[client], qualname_services)

    def test_callable_in_value_iterable_callable(self):
        def client():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client = _qualname(client)
        self.lenient.__dict__[qualname_client] = None
        qualname_services = (_qualname(service1), _qualname(service2))
        self.lenient[client] = (service1, service2)
        self.assertEqual(self.lenient.__dict__[qualname_client], qualname_services)

    def test_active_iterable(self):
        active = "active"
        self.assertIsNone(self.lenient.__dict__[active])
        emsg = "Invalid .* option 'active'"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient[active] = (None,)

    def test_active_str(self):
        active = "active"
        client = "client1"
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient[active] = client
        self.assertEqual(self.lenient.__dict__[active], client)

    def test_active_callable(self):
        def client():
            pass

        active = "active"
        qualname_client = _qualname(client)
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient[active] = client
        self.assertEqual(self.lenient.__dict__[active], qualname_client)

    def test_enable(self):
        enable = "enable"
        self.assertEqual(self.lenient.__dict__[enable], _LENIENT_ENABLE_DEFAULT)
        self.lenient[enable] = True
        self.assertTrue(self.lenient.__dict__[enable])
        self.lenient[enable] = False
        self.assertFalse(self.lenient.__dict__[enable])

    def test_enable_invalid(self):
        emsg = "Invalid .* option 'enable'"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient["enable"] = None


class Test_context(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()
        self.default = dict(active=None, enable=_LENIENT_ENABLE_DEFAULT)

    def copy(self):
        return self.lenient.__dict__.copy()

    def test_nop(self):
        pre = self.copy()
        with self.lenient.context():
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        self.assertEqual(context, self.default)
        self.assertEqual(post, self.default)

    def test_active_str(self):
        client = "client"
        pre = self.copy()
        with self.lenient.context(active=client):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active=client))
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)

    def test_active_callable(self):
        def client():
            pass

        pre = self.copy()
        with self.lenient.context(active=client):
            context = self.copy()
        post = self.copy()
        qualname_client = _qualname(client)
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active=qualname_client))
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)

    def test_kwargs(self):
        client = "client"
        self.lenient.__dict__["service1"] = False
        self.lenient.__dict__["service2"] = False
        pre = self.copy()
        with self.lenient.context(active=client, service1=True, service2=True):
            context = self.copy()
        post = self.copy()
        self.default.update(dict(service1=False, service2=False))
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active=client, service1=True, service2=True))
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)

    def test_args_str(self):
        client = "client"
        services = ("service1", "service2")
        pre = self.copy()
        with self.lenient.context(*services, active=client):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active=client, client=services))
        self.assertEqual(context["active"], expected["active"])
        self.assertEqual(set(context["client"]), set(expected["client"]))
        self.assertEqual(post, self.default)

    def test_args_callable(self):
        def service1():
            pass

        def service2():
            pass

        client = "client"
        services = (service1, service2)
        pre = self.copy()
        with self.lenient.context(*services, active=client):
            context = self.copy()
        post = self.copy()
        qualname_services = tuple([_qualname(service) for service in services])
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active=client, client=qualname_services))
        self.assertEqual(context["active"], expected["active"])
        self.assertEqual(set(context["client"]), set(expected["client"]))
        self.assertEqual(post, self.default)

    def test_context_runtime(self):
        services = ("service1", "service2")
        pre = self.copy()
        with self.lenient.context(*services):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        expected = self.default.copy()
        expected.update(dict(active="__context", __context=services))
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)


class Test_enable(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_getter(self):
        self.assertEqual(self.lenient.enable, _LENIENT_ENABLE_DEFAULT)

    def test_setter_invalid(self):
        emsg = "Invalid .* option 'enable'"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.enable = 0

    def test_setter(self):
        self.assertEqual(self.lenient.enable, _LENIENT_ENABLE_DEFAULT)
        self.lenient.enable = False
        self.assertFalse(self.lenient.enable)


class Test_register_client(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_not_protected(self):
        emsg = "Cannot register .* client"
        for protected in _LENIENT_PROTECTED:
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.register_client(protected, "service")

    def test_str_service_str(self):
        client = "client"
        services = "service"
        self.lenient.register_client(client, services)
        self.assertIn(client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[client], (services,))

    def test_str_services_str(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.register_client(client, services)
        self.assertIn(client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[client], services)

    def test_callable_service_callable(self):
        def client():
            pass

        def service():
            pass

        qualname_client = _qualname(client)
        qualname_service = _qualname(service)
        self.lenient.register_client(client, service)
        self.assertIn(qualname_client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[qualname_client], (qualname_service,))

    def test_callable_services_callable(self):
        def client():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client = _qualname(client)
        qualname_services = (_qualname(service1), _qualname(service2))
        self.lenient.register_client(client, (service1, service2))
        self.assertIn(qualname_client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[qualname_client], qualname_services)

    def test_services_empty(self):
        emsg = "Require at least one .* client service."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.register_client("client", ())

    def test_services_overwrite(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.__dict__[client] = services
        self.assertEqual(self.lenient[client], services)
        new_services = ("service3", "service4")
        self.lenient.register_client(client, services=new_services)
        self.assertEqual(self.lenient[client], new_services)

    def test_services_append(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.__dict__[client] = services
        self.assertEqual(self.lenient[client], services)
        new_services = ("service3", "service4")
        self.lenient.register_client(client, services=new_services, append=True)
        expected = set(services + new_services)
        self.assertEqual(set(self.lenient[client]), expected)


class Test_register_service(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_str(self):
        service = "service"
        self.assertNotIn(service, self.lenient.__dict__)
        self.lenient.register_service(service)
        self.assertIn(service, self.lenient.__dict__)
        self.assertFalse(isinstance(self.lenient.__dict__[service], Iterable))
        self.assertTrue(self.lenient.__dict__[service])

    def test_callable(self):
        def service():
            pass

        qualname_service = _qualname(service)
        self.assertNotIn(qualname_service, self.lenient.__dict__)
        self.lenient.register_service(service)
        self.assertIn(qualname_service, self.lenient.__dict__)
        self.assertFalse(isinstance(self.lenient.__dict__[qualname_service], Iterable))
        self.assertTrue(self.lenient.__dict__[qualname_service])

    def test_not_protected(self):
        emsg = "Cannot register .* service"
        for protected in _LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.register_service("active")


class Test_unregister_client(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_not_protected(self):
        emsg = "Cannot unregister .* client, as .* is a protected .* option."
        for protected in _LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.unregister_client(protected)

    def test_not_in(self):
        emsg = "Cannot unregister unknown .* client"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client("client")

    def test_not_client(self):
        client = "client"
        self.lenient.__dict__[client] = True
        emsg = "Cannot unregister .* client, as .* is not a valid .* client."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client(client)

    def test_not_client_callable(self):
        def client():
            pass

        qualname_client = _qualname(client)
        self.lenient.__dict__[qualname_client] = True
        emsg = "Cannot unregister .* client, as .* is not a valid .* client."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client(client)

    def test_str(self):
        client = "client"
        self.lenient.__dict__[client] = (None,)
        self.lenient.unregister_client(client)
        self.assertNotIn(client, self.lenient.__dict__)

    def test_callable(self):
        def client():
            pass

        qualname_client = _qualname(client)
        self.lenient.__dict__[qualname_client] = (None,)
        self.lenient.unregister_client(client)
        self.assertNotIn(qualname_client, self.lenient.__dict__)


class Test_unregister_service(tests.IrisTest):
    def setUp(self):
        self.lenient = _Lenient()

    def test_not_protected(self):
        emsg = "Cannot unregister .* service, as .* is a protected .* option."
        for protected in _LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.unregister_service(protected)

    def test_not_in(self):
        emsg = "Cannot unregister unknown .* service"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service("service")

    def test_not_service(self):
        service = "service"
        self.lenient.__dict__[service] = (None,)
        emsg = "Cannot unregister .* service, as .* is not a valid .* service."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service(service)

    def test_not_service_callable(self):
        def service():
            pass

        qualname_service = _qualname(service)
        self.lenient.__dict__[qualname_service] = (None,)
        emsg = "Cannot unregister .* service, as .* is not a valid .* service."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service(service)

    def test_str(self):
        service = "service"
        self.lenient.__dict__[service] = True
        self.lenient.unregister_service(service)
        self.assertNotIn(service, self.lenient.__dict__)

    def test_callable(self):
        def service():
            pass

        qualname_service = _qualname(service)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.unregister_service(service)
        self.assertNotIn(qualname_service, self.lenient.__dict__)


if __name__ == "__main__":
    tests.main()
