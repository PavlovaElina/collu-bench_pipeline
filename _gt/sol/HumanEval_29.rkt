#lang racket
(define (filter_by_prefix strings prefix)
  (filter (lambda (s) (string-prefix? s prefix)) strings))
