#lang racket
(define (filter_by_substring strings substring)
  (filter (lambda (s) (string-contains? s substring)) strings))
